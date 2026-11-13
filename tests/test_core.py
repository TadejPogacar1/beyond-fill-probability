from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from fill_probability.config import ExperimentConfig, LONG_FEATURES
from fill_probability.labeling import crossing_fill_times
from fill_probability.simulator import simulate_single_level_day


class LabelTests(unittest.TestCase):
    def test_crossed_through_rule_is_strict(self):
        result = crossing_fill_times(
            np.array([99], dtype=np.int32),
            np.array([0], dtype=np.int64),
            np.array([1.0], dtype=np.float64),
            5_000,
            np.array([99, 98, 101], dtype=np.int32),
            np.array([1_000, 2_000, 3_000], dtype=np.int64),
            np.array([1.0, 1.0, 1.0], dtype=np.float64),
        )
        self.assertEqual(int(result[0]), 2_000)

    def test_horizon_prevents_late_fill(self):
        result = crossing_fill_times(
            np.array([99], dtype=np.int32),
            np.array([0], dtype=np.int64),
            np.array([1.0], dtype=np.float64),
            1_500,
            np.array([99, 98], dtype=np.int32),
            np.array([1_000, 2_000], dtype=np.int64),
            np.array([1.0, 1.0], dtype=np.float64),
        )
        self.assertEqual(int(result[0]), -1)

    def test_preplacement_trades_are_never_used(self):
        result = crossing_fill_times(
            np.array([99], dtype=np.int32),
            np.array([5_000], dtype=np.int64),
            np.array([0.0], dtype=np.float64),
            10_000,
            np.array([99], dtype=np.int32),
            np.array([1_000], dtype=np.int64),
            np.array([1.0], dtype=np.float64),
        )
        self.assertEqual(int(result[0]), -1)

    def test_fill_must_be_strictly_after_placement(self):
        result = crossing_fill_times(
            np.array([99], dtype=np.int32),
            np.array([5_000], dtype=np.int64),
            np.array([0.0], dtype=np.float64),
            10_000,
            np.array([99, 99], dtype=np.int32),
            np.array([5_000, 6_000], dtype=np.int64),
            np.array([1.0, 1.0], dtype=np.float64),
        )
        self.assertEqual(int(result[0]), 6_000)

    def test_integer_quantities_preserve_strict_queue_rule(self):
        result = crossing_fill_times(
            np.array([99], dtype=np.int32),
            np.array([0], dtype=np.int64),
            np.array([0.3], dtype=np.float64),
            10_000,
            np.array([99, 99, 99], dtype=np.int32),
            np.array([1_000, 2_000, 3_000], dtype=np.int64),
            np.array([0.1, 0.2, 0.01], dtype=np.float64),
        )
        self.assertEqual(int(result[0]), 3_000)


class SimulatorTests(unittest.TestCase):
    def test_buy_then_end_of_day_liquidation(self):
        config = replace(
            ExperimentConfig(),
            order_size_btc=0.01,
            maker_fee_bps=0.0,
            fill_slippage_bps=0.0,
            liquidation_slippage_bps=0.0,
            inventory_skew_bps=0.0,
        )
        frame = pd.DataFrame(
            {
                "timestamp_ms": [0],
                "side": [1],
                "depth_bps": [2.0],
                "order_price": [100.0],
                "fill_time_ms": [1_000],
                "mid_t": [100.0],
            }
        )
        result = simulate_single_level_day(frame, np.array([1.0]), 110.0, config)
        self.assertAlmostEqual(result["pnl_usd"], 0.10, places=8)
        self.assertEqual(result["fills"], 1)

    def test_nonpositive_scores_do_not_submit_orders(self):
        config = replace(ExperimentConfig(), inventory_skew_bps=0.0)
        frame = pd.DataFrame(
            {
                "timestamp_ms": [0],
                "side": [1],
                "depth_bps": [2.0],
                "order_price": [100.0],
                "fill_time_ms": [1_000],
                "mid_t": [100.0],
            }
        )
        result = simulate_single_level_day(frame, np.array([0.0]), 100.0, config)
        self.assertEqual(result["submitted_orders"], 0)
        self.assertEqual(result["fills"], 0)
        self.assertAlmostEqual(result["pnl_usd"], 0.0)


class FeatureTests(unittest.TestCase):
    def test_long_feature_count_matches_paper(self):
        self.assertEqual(len(LONG_FEATURES), 28)


if __name__ == "__main__":
    unittest.main()
