# Methodology

## Synthetic passive orders

The L2 book is reconstructed from snapshots and incremental updates, then sampled
once per second. A synthetic bid and ask are evaluated at 12 depths from the
prevailing midprice:

```text
0.1, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30 bps
```

Each order is labeled at 60, 300, and 600 second horizons.

## Causal crossed-through label

L2 data provide displayed quantity but not individual FIFO order identities.
Queue ahead is approximated by displayed same-side quantity at the simulated
price when the order is placed.

For a bid at price `P_i`, placement time `t_i`, queue ahead `q_i`, and horizon
`T`, the fill indicator is

```math
\mathrm{fill}_{i,T} =
\mathbf{1}\left(
\sum_{t_i < t \le t_i+T}
\mathbf{1}\{P_t^{trade} \le P_i\}v_t > q_i
\right).
```

The price inequality is reversed for asks. Quantities are converted to integer
satoshis in a Fenwick tree. This prevents floating-point prefix errors and makes
pre-placement trades impossible to use.

## Features

- 10 base features: depth, side, spread, imbalance, top-of-book size, queue ahead,
  and log transforms;
- 17 short-context features: base features plus recent return, volatility, and
  interaction terms;
- 28 long-context features: short features plus 5/30 minute state, time-of-day,
  session volatility, running range, and interactions.

## Four objectives

Let `p_i` be predicted fill probability, `d_i` depth, `c` explicit cost, and
`AS_i` predicted adverse selection.

### Fill only

```math
s_i = \hat p_i(d_i-c).
```

### Two stage

```math
s_i = \hat p_i(d_i-\widehat{AS}_i-c).
```

### Unified short and unified long

Both predict realized order-level net outcome directly:

```math
y_i = \mathbf{1}\{\mathrm{fill}_i\}
      (\mathrm{gross\ outcome}_i-c).
```

Non-filled orders receive a zero target. Unified short uses 17 features and
unified long uses 28.

## Simulator

At most one bid and one ask can be live. Orders cannot be canceled. A fill or
timeout starts a 10 second cooldown on that side. The highest inventory-adjusted
score among the 12 depths is submitted only when it exceeds zero.

| Parameter | Value |
|---|---:|
| Order size | 0.01 BTC |
| Inventory cap | +/-0.10 BTC |
| Inventory skew at cap | 8 bps |
| Maker fee | 1.0 bps per fill |
| Fill slippage | 0.5 bps per fill |
| End-of-day liquidation slippage | 2.0 bps |

## Frozen holdout

The four model components are fitted once for each horizon on 42 earlier days.
All training rows are used. Model weights and the zero-bps decision threshold are
then frozen for seven later days. No holdout row enters model fitting or threshold
selection.
