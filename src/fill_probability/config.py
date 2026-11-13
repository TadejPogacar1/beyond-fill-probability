from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_FEATURES = [
    "depth_bps",
    "log_depth",
    "side",
    "imbalance",
    "abs_imbalance",
    "relative_spread_bps",
    "l1_size",
    "log_l1_size",
    "queue_ahead",
    "log_queue_ahead",
]

SHORT_FEATURES = BASE_FEATURES + [
    "ret_10s_bps",
    "ret_30s_bps",
    "ret_60s_bps",
    "vol_30s_bps",
    "vol_60s_bps",
    "momentum_x_side",
    "depth_x_volatility",
]

LONG_FEATURES = SHORT_FEATURES + [
    "ret_300s_bps",
    "ret_1800s_bps",
    "vol_300s_bps",
    "vol_1800s_bps",
    "micro_macro_vol_ratio",
    "momentum_acceleration",
    "hour_sin",
    "hour_cos",
    "session_volatility_bps",
    "running_daily_range_bps",
    "depth_x_5m_volatility",
]

ALL_DEPTHS_BPS = (
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    7.0,
    10.0,
    15.0,
    20.0,
    30.0,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Assumptions that materially affect labels, models, and simulation."""

    raw_root: Path = Path("data/raw/cryptohftdata/binance_spot/BTCUSDT")
    state_dir: Path = Path("data/processed/market_state_1s")
    sample_dir: Path = Path("data/processed/samples/T300")
    result_dir: Path = Path("results/generated")
    audit_json: Path | None = None

    symbol: str = "BTCUSDT"
    price_scale: int = 100
    max_price_usd: int = 200_000

    all_depths_bps: tuple[float, ...] = ALL_DEPTHS_BPS
    final_depths_bps: tuple[float, ...] = ALL_DEPTHS_BPS
    horizon_seconds: int = 300
    decision_cadence_seconds: int = 1

    order_size_btc: float = 0.01
    inventory_cap_btc: float = 0.10
    inventory_skew_bps: float = 8.0
    maker_fee_bps: float = 1.0
    fill_slippage_bps: float = 0.5
    liquidation_slippage_bps: float = 2.0
    cooldown_seconds: int = 10
    score_threshold_bps: float = 0.0

    annualization_days: int = 365
    max_train_rows_per_day: int = 100_000
    random_seed: int = 202604

    max_iter: int = 100
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 50
    l2_regularization: float = 1.0

    development_dates: tuple[str, ...] = tuple(
        f"2026-08-{day:02d}" for day in range(2, 17)
    )
    oos_dates: tuple[str, ...] = tuple(
        f"2026-08-{day:02d}" for day in range(17, 23)
    )

    @property
    def paper_style_dates(self) -> tuple[str, ...]:
        return self.development_dates + self.oos_dates

    @property
    def explicit_cost_bps(self) -> float:
        return self.maker_fee_bps + self.fill_slippage_bps

    @property
    def max_price_tick(self) -> int:
        return self.max_price_usd * self.price_scale


def depth_token(depth: float) -> str:
    text = f"{depth:g}".replace(".", "p")
    return f"d_{text}"
