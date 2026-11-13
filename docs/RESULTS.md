# Seven-Day Holdout Results

## All configurations

| Horizon | Objective | Total PnL | Mean/day | Sharpe | Wins | Fills/day | PnL/fill | Max drawdown |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 60 | Fill only | -$577.44 | -$82.49 | -31.39 | 0/7 | 346.86 | -$0.238 | -$577.44 |
| 60 | Two stage | -$52.99 | -$7.57 | -10.18 | 2/7 | 5.86 | -$1.293 | -$54.05 |
| 60 | Unified short | -$45.68 | -$6.53 | -9.55 | 1/7 | 9.43 | -$0.692 | -$47.30 |
| 60 | Unified long | -$11.77 | -$1.68 | -3.44 | 3/7 | 19.43 | -$0.087 | -$24.98 |
| 300 | Fill only | -$185.79 | -$26.54 | -4.82 | 3/7 | 151.43 | -$0.175 | -$255.04 |
| 300 | Two stage | -$130.66 | -$18.67 | -6.33 | 3/7 | 25.71 | -$0.726 | -$195.09 |
| 300 | Unified short | -$60.97 | -$8.71 | -3.46 | 5/7 | 20.57 | -$0.423 | -$110.47 |
| 300 | Unified long | -$101.22 | -$14.46 | -5.19 | 4/7 | 20.86 | -$0.693 | -$128.20 |
| 600 | Fill only | -$256.00 | -$36.57 | -6.25 | 3/7 | 91.29 | -$0.401 | -$270.01 |
| 600 | Two stage | -$99.45 | -$14.21 | -3.27 | 4/7 | 24.86 | -$0.572 | -$217.54 |
| 600 | Unified short | -$34.42 | -$4.92 | -1.07 | 4/7 | 27.71 | -$0.177 | -$162.36 |
| **600** | **Unified long** | **+$43.87** | **+$6.27** | **2.32** | **6/7** | **30.29** | **+$0.207** | **-$105.24** |

Annualized Sharpe is included for completeness but is not stable with seven daily
observations.

## Best configuration by day

| Date | PnL | Fills | Submitted orders | Ending inventory before liquidation |
|---|---:|---:|---:|---:|
| 2026-09-13 | +$3.04 | 6 | 93 | +0.06 BTC |
| 2026-09-14 | +$29.50 | 14 | 114 | 0.00 BTC |
| 2026-09-15 | -$105.24 | 75 | 175 | +0.09 BTC |
| 2026-09-16 | +$41.01 | 30 | 93 | +0.06 BTC |
| 2026-09-17 | +$6.12 | 13 | 141 | +0.03 BTC |
| 2026-09-18 | +$25.64 | 59 | 151 | +0.05 BTC |
| 2026-09-19 | +$43.79 | 15 | 169 | -0.09 BTC |

The positive result is driven by six modest winning days and one substantial
losing day. It supports direct net-outcome ranking within this window, but the
sample is too short to establish robustness.

## Interpretation

- Fill-only trading is consistently weak and excessively active.
- Direct net-outcome prediction is more selective.
- Longer context is useful at the 600 second horizon in this window.
- Inventory exposure remains capable of producing a large daily loss.
- A later extended holdout did not preserve profitability.
