"""Tests for Task 1, data processing (owner: Lim Yon), and the error measures."""

import unittest

import numpy as np
import pandas as pd

from tests.helpers import config, prepared
from tbrgs import metrics
from tbrgs.data import (INTERVAL, SeriesScaler, load_raw, make_windows, parse_location, split_dates,
                        time_features)


class TestLoading(unittest.TestCase):
    def setUp(self):
        self.data = prepared().data

    def test_sizes(self):
        self.assertEqual(self.data.flows.shape, (31 * 96, 140))
        self.assertEqual(len(self.data.sites), 40)
        self.assertEqual(self.data.flows.index[0], pd.Timestamp("2006-10-01 00:00"))
        self.assertEqual(self.data.flows.index[-1], pd.Timestamp("2006-10-31 23:45"))

    def test_site_4335_keeps_both_detectors(self):
        detectors = self.data.detectors
        same_name = detectors[detectors["location"] == "HIGH_ST NE of CHARLES_ST"]
        self.assertEqual(sorted(same_name.index), ["4335-2", "4335-6"])
        # They are different series, not a duplicate.
        a, b = self.data.flows["4335-2"], self.data.flows["4335-6"]
        self.assertFalse(np.allclose(a.fillna(0), b.fillna(0)))

    def test_zero_coordinates_replaced(self):
        row = self.data.detectors.loc["4266-1"]  # AUBURN_RD N of BURWOOD_RD had 0, 0
        self.assertAlmostEqual(row["latitude"], -37.825, places=2)
        self.assertAlmostEqual(row["longitude"], 145.043, places=2)

    def test_missing_days_are_nan_not_zero(self):
        series = self.data.flows["3001-6"]  # CHURCH_ST SW of BARKERS_RD: 2 days of data
        self.assertEqual(int(series.notna().sum()), 2 * 96)
        self.assertEqual(int(self.data.detectors.loc["3001-6", "days"]), 2)

    def test_interval_columns_line_up_with_times(self):
        raw = load_raw(config().path_of("data", "scats_file"))
        row = raw[(raw["SCATS Number"] == 2000) & (raw["VR Internal Loc"] == 1)].iloc[3]  # 4 Oct
        day = pd.Timestamp(row["Date"]).normalize()
        detector = "2000-1"
        self.assertEqual(self.data.flows.at[day, detector], row["V00"])
        self.assertEqual(self.data.flows.at[day + 33 * INTERVAL, detector], row["V33"])
        self.assertEqual(self.data.flows.at[day + 95 * INTERVAL, detector], row["V95"])

    def test_site_names_follow_the_brief(self):
        self.assertEqual(self.data.sites.loc[2000, "name"], "WARRIGAL_RD/TOORAK_RD")
        self.assertEqual(self.data.sites.loc[3002, "name"], "DENMARK_ST/BARKERS_RD")

    def test_parse_location(self):
        self.assertEqual(parse_location("WARRIGAL_RD N of TOORAK_RD"), ("WARRIGAL_RD", "N", "TOORAK_RD"))
        self.assertEqual(parse_location("BALWYN_RD N OF BELMORE_RD"), ("BALWYN_RD", "N", "BELMORE_RD"))
        self.assertIsNone(parse_location("no direction here"))


class TestSplitAndScaling(unittest.TestCase):
    def setUp(self):
        self.p = prepared()

    def test_split_is_chronological_and_disjoint(self):
        s = self.p.split
        self.assertLess(s.train[1], s.validation[0])
        self.assertLess(s.validation[1], s.test[0])
        self.assertEqual(s.test[0], pd.Timestamp("2006-10-25"))
        self.assertEqual(s.test[1], pd.Timestamp("2006-10-31 23:45"))

    def test_bad_split_rejected(self):
        with self.assertRaises(ValueError):
            split_dates(self.p.data.flows, "2006-10-25", "2006-10-20")

    def test_scaler_uses_training_period_only(self):
        flows = self.p.data.flows
        for detector in ("2000-1", "3002-2", "4335-6"):
            if detector in flows:
                expected = flows.loc[:self.p.split.train[1], detector].max()
                self.assertEqual(self.p.scaler.scales[detector], expected)

    def test_scaler_round_trip(self):
        values = np.array([0.0, 0.5, 1.0])
        back = self.p.scaler.inverse(values, ["2000-1"] * 3)
        np.testing.assert_allclose(back / self.p.scaler.scales["2000-1"], values)

    def test_scaler_handles_all_zero_series(self):
        frame = pd.DataFrame({"a": [0.0, 0.0], "b": [1.0, 3.0]},
                             index=pd.date_range("2006-10-01", periods=2, freq="15min"))
        scaler = SeriesScaler().fit(frame, frame.index[-1])
        self.assertEqual(scaler.scales["a"], 1.0)
        self.assertEqual(scaler.scales["b"], 3.0)


class TestWindows(unittest.TestCase):
    def setUp(self):
        self.p = prepared()

    def test_window_inputs_and_target_line_up(self):
        test = self.p.windows["test"]
        scaled = self.p.scaled
        for i in (0, 1000, len(test) - 1):
            t, detector = pd.Timestamp(test.times[i]), test.detectors[i]
            self.assertAlmostEqual(float(test.y[i]), scaled.at[t, detector], places=5)
            for lag in range(1, self.p.lags + 1):
                self.assertAlmostEqual(float(test.X[i, -lag, 0]), scaled.at[t - lag * INTERVAL, detector], places=5)

    def test_no_missing_values_in_windows(self):
        for part in ("train", "validation", "test"):
            w = self.p.windows[part]
            self.assertFalse(np.isnan(w.X).any())
            self.assertFalse(np.isnan(w.y).any())

    def test_targets_stay_inside_their_part(self):
        for part in ("train", "validation", "test"):
            start, end = self.p.split.part(part)
            times = pd.DatetimeIndex(self.p.windows[part].times)
            self.assertGreaterEqual(times.min(), start)
            self.assertLessEqual(times.max(), end)

    def test_gap_days_produce_no_windows(self):
        windows = make_windows(self.p.scaled, 12, detectors=["3001-6"])
        self.assertLessEqual(len(windows), 2 * 96 - 12)

    def test_time_features(self):
        features = time_features(pd.to_datetime(["2006-10-01 00:00", "2006-10-02 06:00"]))
        np.testing.assert_allclose(features[0], [0.0, 1.0, 1.0], atol=1e-9)   # Sunday midnight
        np.testing.assert_allclose(features[1], [1.0, 0.0, 0.0], atol=1e-9)   # Monday 06:00

    def test_flat_features_shape(self):
        w = self.p.windows["validation"]
        self.assertEqual(w.flat().shape, (len(w), self.p.lags + 3))


class TestMetrics(unittest.TestCase):
    def test_simple_values(self):
        actual, predicted = [10, 20, 30], [12, 18, 30]
        self.assertAlmostEqual(metrics.mae(actual, predicted), 4 / 3)
        self.assertAlmostEqual(metrics.rmse(actual, predicted), np.sqrt(8 / 3))
        self.assertAlmostEqual(metrics.nrmse(actual, predicted), np.sqrt(8 / 3) / 20)
        self.assertAlmostEqual(metrics.r2(actual, actual), 1.0)

    def test_mape_ignores_tiny_flows(self):
        self.assertAlmostEqual(metrics.mape([1, 100], [50, 110]), 10.0)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            metrics.mae([1, 2], [1])


if __name__ == "__main__":
    unittest.main()
