"""Tests for Task 2, the models (LSTM: Lim Yon, GRU: Jonathan Yu, XGBoost: Lau Ka Meng).

The models are trained here on a small synthetic data set (a clean daily
pattern) so the tests run in seconds and check behaviour, not accuracy.
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from tests.helpers import trained, config, prepared
from tbrgs.data import make_windows
from tbrgs.models import load_model, model_class
from tbrgs.models.baselines import HistoricalAverageModel, PersistenceModel


def synthetic_windows(lags=6):
    """Three detectors with a smooth daily cycle, 10 days long, already scaled to 0-1."""
    times = pd.date_range("2006-10-01", periods=10 * 96, freq="15min")
    hours = times.hour + times.minute / 60
    base = 0.5 - 0.4 * np.cos(2 * np.pi * hours / 24)
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({f"d{i}": base * (0.6 + 0.2 * i) + rng.normal(0, 0.01, len(times)) for i in range(3)},
                         index=times)
    train = make_windows(frame, lags, times[0], times[7 * 96 - 1])
    validation = make_windows(frame, lags, times[7 * 96], times[-1])
    return train, validation


FAST = {"lstm": {"units": 8, "layers": 1, "epochs": 3, "batch_size": 64},
        "gru": {"units": 8, "layers": 1, "epochs": 3, "batch_size": 64},
        "xgboost": {"n_estimators": 60, "max_depth": 4, "early_stopping_rounds": 10}}


class ModelBehaviour:
    """Checks every model must pass; mixed into one TestCase per model."""
    name = None

    @classmethod
    def setUpClass(cls):
        cls.train, cls.validation = synthetic_windows()
        cls.model = model_class(cls.name)(cls.train.lags, FAST[cls.name])
        cls.meta = cls.model.fit(cls.train, cls.validation, log=lambda line: None)

    def test_learns_something(self):
        predicted = self.model.predict_windows(self.validation)
        error = np.mean(np.abs(predicted - self.validation.y))
        spread = np.mean(np.abs(self.validation.y - self.validation.y.mean()))
        self.assertLess(error, spread)  # better than always predicting the mean

    def test_prediction_shape_and_sign(self):
        predicted = self.model.predict_windows(self.validation)
        self.assertEqual(predicted.shape, (len(self.validation),))
        self.assertTrue((predicted >= 0).all())

    def test_save_and_load_give_the_same_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            self.model.save(folder)
            again = type(self.model).load(folder)
            np.testing.assert_allclose(again.predict_windows(self.validation),
                                       self.model.predict_windows(self.validation), rtol=1e-5, atol=1e-6)
            self.assertEqual(again.lags, self.model.lags)

    def test_wrong_number_of_lags_rejected(self):
        with self.assertRaises(ValueError):
            self.model.predict(np.zeros((2, self.model.lags + 1, 4)), [pd.Timestamp("2006-10-01")] * 2)

    def test_metadata_recorded(self):
        self.assertIn("train_seconds", self.meta)
        self.assertEqual(self.meta["lags"], self.train.lags)


class TestLSTM(ModelBehaviour, unittest.TestCase):
    name = "lstm"

    def test_has_two_lstm_layers_by_default(self):
        model = model_class("lstm")(12)
        network = model.build()
        kinds = [layer.__class__.__name__ for layer in network.layers]
        self.assertEqual(kinds.count("LSTM"), 2)


class TestGRU(ModelBehaviour, unittest.TestCase):
    name = "gru"

    def test_fewer_weights_than_lstm(self):
        lstm = model_class("lstm")(12).build().count_params()
        gru = model_class("gru")(12).build().count_params()
        self.assertLess(gru, lstm)


class TestXGBoost(ModelBehaviour, unittest.TestCase):
    name = "xgboost"


class TestBaselines(unittest.TestCase):
    def test_persistence_repeats_last_value(self):
        train, _ = synthetic_windows()
        np.testing.assert_array_equal(PersistenceModel().predict_windows(train), train.X[:, -1, 0])

    def test_historical_average(self):
        times = pd.date_range("2006-10-02", periods=96 * 2, freq="15min")  # Monday and Tuesday
        frame = pd.DataFrame({"d": np.r_[np.full(96, 2.0), np.full(96, 4.0)]}, index=times)
        model = HistoricalAverageModel().fit(frame, times[-1])
        value = model.predict_for(["d"], [pd.Timestamp("2006-10-04 08:00")])  # a Wednesday
        self.assertAlmostEqual(float(value[0]), 3.0)


class TestUnknownModel(unittest.TestCase):
    def test_unknown_name(self):
        with self.assertRaises(ValueError):
            model_class("random_forest")

    def test_missing_model_files(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                load_model("lstm", folder)


@unittest.skipUnless(all(trained(n) for n in ("lstm", "gru", "xgboost")), "train the models first")
class TestTrainedModels(unittest.TestCase):
    """The saved models must beat the persistence baseline on the test week."""

    def test_beat_persistence(self):
        p = prepared()
        test = p.windows["test"]
        actual = p.scaler.inverse(test.y, test.detectors)
        baseline = np.mean(np.abs(actual - p.scaler.inverse(PersistenceModel().predict_windows(test),
                                                            test.detectors)))
        for name in ("lstm", "gru", "xgboost"):
            model = load_model(name, config().path_of("paths", "model_dir"))
            error = np.mean(np.abs(actual - p.scaler.inverse(model.predict_windows(test), test.detectors)))
            self.assertLess(error, baseline, name)


if __name__ == "__main__":
    unittest.main()
