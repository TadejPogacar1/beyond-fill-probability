from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ExperimentConfig


@dataclass
class LiveOrder:
    side: int
    placement_ms: int
    expiry_ms: int
    fill_time_ms: int
    price: float
    raw_score_bps: float
    adjusted_score_bps: float


def simulate_single_level_day(
    candidates: pd.DataFrame,
    predicted_score_bps: np.ndarray,
    end_of_day_mid: float,
    config: ExperimentConfig,
) -> dict:
    """Run the paper-style IDLE/LIVE/COOLDOWN single-level simulator.

    The paper does not publish its exact inventory-skew equation.  This
    implementation applies the stated 8 bps-at-limit magnitude to the entry
    score: inventory-increasing orders are penalized and inventory-reducing
    orders receive an equal bonus.
    """

    frame = candidates[
        [
            "timestamp_ms",
            "side",
            "depth_bps",
            "order_price",
            "fill_time_ms",
            "mid_t",
        ]
    ].copy()
    frame["score_bps"] = np.asarray(predicted_score_bps, dtype=np.float64)
    frame = frame.sort_values(["timestamp_ms", "side", "depth_bps"], kind="stable")
    frame = frame.reset_index(drop=True)

    cash = 0.0
    inventory = 0.0
    max_abs_inventory = 0.0
    fills = 0
    timeouts = 0
    submitted = 0
    live: dict[int, LiveOrder | None] = {1: None, -1: None}
    cooldown_until = {1: -1, -1: -1}

    def finish_due_orders(now_ms: int) -> None:
        nonlocal cash, inventory, fills, timeouts, max_abs_inventory
        for side in (1, -1):
            order = live[side]
            if order is None:
                continue
            filled = order.placement_ms < order.fill_time_ms <= order.expiry_ms
            event_ms = order.fill_time_ms if filled else order.expiry_ms
            if event_ms > now_ms:
                continue
            if filled:
                notional = config.order_size_btc * order.price
                cost = notional * config.explicit_cost_bps / 10_000.0
                if side == 1:
                    cash -= notional + cost
                    inventory += config.order_size_btc
                else:
                    cash += notional - cost
                    inventory -= config.order_size_btc
                fills += 1
                max_abs_inventory = max(max_abs_inventory, abs(inventory))
            else:
                timeouts += 1
            live[side] = None
            cooldown_until[side] = event_ms + config.cooldown_seconds * 1000

    timestamps = frame["timestamp_ms"].to_numpy(np.int64)
    boundaries = np.flatnonzero(
        np.r_[True, timestamps[1:] != timestamps[:-1], True]
    )
    for boundary_index in range(boundaries.size - 1):
        start = boundaries[boundary_index]
        end = boundaries[boundary_index + 1]
        now_ms = int(timestamps[start])
        finish_due_orders(now_ms)
        group = frame.iloc[start:end]

        for side in (1, -1):
            if live[side] is not None or now_ms < cooldown_until[side]:
                continue
            proposed_inventory = inventory + side * config.order_size_btc
            if abs(proposed_inventory) > config.inventory_cap_btc + 1e-12:
                continue
            side_group = group.loc[group["side"] == side]
            if side_group.empty:
                continue
            inventory_ratio = inventory / config.inventory_cap_btc
            adjusted = (
                side_group["score_bps"].to_numpy(np.float64)
                - inventory_ratio * side * config.inventory_skew_bps
            )
            best_local = int(np.argmax(adjusted))
            best_adjusted = float(adjusted[best_local])
            if not np.isfinite(best_adjusted) or best_adjusted <= config.score_threshold_bps:
                continue
            row = side_group.iloc[best_local]
            live[side] = LiveOrder(
                side=side,
                placement_ms=now_ms,
                expiry_ms=now_ms + config.horizon_seconds * 1000,
                fill_time_ms=int(row["fill_time_ms"]),
                price=float(row["order_price"]),
                raw_score_bps=float(row["score_bps"]),
                adjusted_score_bps=best_adjusted,
            )
            submitted += 1

    day_start = int(pd.Timestamp(frame.iloc[0]["timestamp_ms"], unit="ms", tz="UTC").normalize().value // 1_000_000)
    day_end = day_start + 86_400_000
    finish_due_orders(day_end)

    inventory_before_liquidation = inventory
    liquidation_cost = abs(inventory) * end_of_day_mid * config.liquidation_slippage_bps / 10_000.0
    if inventory > 0:
        cash += inventory * end_of_day_mid - liquidation_cost
    elif inventory < 0:
        cash -= (-inventory) * end_of_day_mid + liquidation_cost
    inventory = 0.0

    return {
        "pnl_usd": float(cash),
        "fills": int(fills),
        "submitted_orders": int(submitted),
        "timeouts": int(timeouts),
        "max_abs_inventory_btc": float(max_abs_inventory),
        "inventory_before_liquidation_btc": float(inventory_before_liquidation),
        "liquidation_cost_usd": float(liquidation_cost),
    }
