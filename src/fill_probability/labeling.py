from __future__ import annotations

import numpy as np
from numba import njit


QUANTITY_SCALE = 100_000_000


@njit(cache=True)
def _fenwick_add(tree: np.ndarray, zero_based_index: int, value: int) -> None:
    index = zero_based_index + 1
    while index < tree.size:
        tree[index] += value
        index += index & -index


@njit(cache=True)
def _fenwick_sum_first(tree: np.ndarray, count: int) -> np.int64:
    total = np.int64(0)
    index = count
    while index > 0:
        total += tree[index]
        index -= index & -index
    return total


@njit(cache=True)
def _first_prefix_strictly_greater(tree: np.ndarray, target: int) -> int:
    """Return the first zero-based position whose prefix sum exceeds target."""

    n = tree.size - 1
    index = 0
    accumulated = np.int64(0)
    bit = 1
    while bit << 1 <= n:
        bit <<= 1
    while bit:
        candidate = index + bit
        if candidate <= n and accumulated + tree[candidate] <= target:
            index = candidate
            accumulated += tree[candidate]
        bit >>= 1
    return index


@njit(cache=True)
def crossing_fill_times(
    candidate_key: np.ndarray,
    candidate_time_ms: np.ndarray,
    queue_ahead: np.ndarray,
    horizon_ms: int,
    trade_key: np.ndarray,
    trade_time_ms: np.ndarray,
    trade_quantity: np.ndarray,
) -> np.ndarray:
    """Label conservative crossed-through fills with a causal Fenwick sweep.

    A trade qualifies when ``trade_key <= candidate_key``. Bid prices are used
    directly; ask prices are negated so both sides use the same comparison.
    Quantities are converted to integer satoshis to avoid floating-point prefix
    errors. A valid fill must occur strictly after placement and no later than
    the requested horizon.
    """

    result = np.full(candidate_key.size, -1, dtype=np.int64)
    if trade_key.size == 0 or candidate_key.size == 0:
        return result

    trade_quantity_int = np.rint(
        np.maximum(trade_quantity, 0.0) * QUANTITY_SCALE
    ).astype(np.int64)
    queue_ahead_int = np.rint(
        np.maximum(queue_ahead, 0.0) * QUANTITY_SCALE
    ).astype(np.int64)

    candidate_order = np.argsort(candidate_key)
    trade_price_order = np.argsort(trade_key)
    tree = np.zeros(trade_key.size + 1, dtype=np.int64)
    added = 0

    for ordered_candidate in candidate_order:
        threshold = candidate_key[ordered_candidate]
        while added < trade_price_order.size:
            trade_index = trade_price_order[added]
            if trade_key[trade_index] > threshold:
                break
            _fenwick_add(tree, trade_index, trade_quantity_int[trade_index])
            added += 1

        placement = candidate_time_ms[ordered_candidate]
        first_after = np.searchsorted(trade_time_ms, placement, side="right")
        baseline = _fenwick_sum_first(tree, first_after)
        target = baseline + queue_ahead_int[ordered_candidate]
        fill_position = _first_prefix_strictly_greater(tree, target)

        if fill_position < trade_time_ms.size:
            fill_ms = trade_time_ms[fill_position]
            if placement < fill_ms <= placement + horizon_ms:
                result[ordered_candidate] = fill_ms
    return result
