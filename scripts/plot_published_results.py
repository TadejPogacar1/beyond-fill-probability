from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "results" / "published"
FIGURES = PUBLISHED / "figures"


def objective_label(value: str) -> str:
    return value.replace("_", " ").title()


def main() -> None:
    summary = pd.read_csv(PUBLISHED / "model_summary.csv")
    daily = pd.read_csv(PUBLISHED / "best_strategy_daily.csv", parse_dates=["date"])
    FIGURES.mkdir(parents=True, exist_ok=True)

    ordered = summary.sort_values("total_pnl_usd", ascending=True).reset_index(drop=True)
    labels = [
        f"{int(row.horizon_seconds)}s | {objective_label(row.objective)}"
        for row in ordered.itertuples()
    ]
    values = ordered["total_pnl_usd"].to_numpy()
    colors = np.where(values > 0, "#2a9d8f", "#607d8b")

    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 6.5))
    day_colors = np.where(daily["pnl_usd"] >= 0, "#2a9d8f", "#d1495b")
    left.bar(daily["date"].dt.strftime("%b %d"), daily["pnl_usd"], color=day_colors)
    left.axhline(0, color="#222222", linewidth=0.8)
    left.set_title("T=600 Unified Long: Daily Net PnL")
    left.set_ylabel("Net PnL (USD)")
    left.tick_params(axis="x", rotation=35)
    for index, value in enumerate(daily["pnl_usd"]):
        left.text(
            index,
            value + (3 if value >= 0 else -7),
            f"${value:,.1f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )

    right.barh(labels, values, color=colors)
    right.axvline(0, color="#222222", linewidth=0.8)
    right.set_title("Frozen Seven-Day Holdout: All Configurations")
    right.set_xlabel("Total net PnL (USD)")
    for index, value in enumerate(values):
        if value >= 0:
            label_x, label_ha, label_color = value - 3, "right", "white"
        elif value <= -80:
            label_x, label_ha, label_color = value + 5, "left", "white"
        else:
            label_x, label_ha, label_color = value - 6, "right", "#222222"
        right.text(
            label_x,
            index,
            f"${value:,.1f}",
            ha=label_ha,
            va="center",
            fontsize=8,
            color=label_color,
        )

    fig.suptitle(
        "Queue-Aware Market Making | 42-Day Training, 7-Day Frozen Holdout",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Net of modeled maker fee, fill slippage, and end-of-day liquidation. "
        "Seven days are insufficient for a stable performance claim.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(FIGURES / "holdout_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
