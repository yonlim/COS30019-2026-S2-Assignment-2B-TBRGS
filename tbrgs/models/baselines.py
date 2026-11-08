"""Two simple reference predictors used in the evaluation (not counted as ML algorithms).

Owner: Lim Yon (105811100)

A learned model is only useful if it beats predictions that need no
learning at all:

    Persistence         "the next 15 minutes will be like the last 15 minutes"
    HistoricalAverage   "the next 15 minutes will be like the average of the
                        same detector, at the same time of day, on the same
                        kind of day (weekday or weekend) in the training weeks"

HistoricalAverage is also the route finder's fallback when a detector has no
recent history to feed a model (for example, a missing day in the data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PersistenceModel:
    name = "persistence"

    def predict_windows(self, windows) -> np.ndarray:
        return np.asarray(windows.X[:, -1, 0], dtype=float)


class HistoricalAverageModel:
    name = "historical"

    def __init__(self):
        self.table = None  # Series indexed by (detector, weekend, slot) -> mean scaled flow

    def fit(self, scaled: pd.DataFrame, end) -> "HistoricalAverageModel":
        part = scaled.loc[:end]
        long = part.stack().rename("flow").reset_index()
        long.columns = ["time", "detector", "flow"]
        times = pd.DatetimeIndex(long["time"])
        long["weekend"] = (times.day_of_week >= 5).astype(int)
        long["slot"] = times.hour * 4 + times.minute // 15
        self.table = long.groupby(["detector", "weekend", "slot"])["flow"].mean()
        # Detector-level mean, used if a (weekend, slot) combination was never observed.
        self.detector_mean = long.groupby("detector")["flow"].mean()
        return self

    def predict_for(self, detectors, times) -> np.ndarray:
        times = pd.DatetimeIndex(times)
        weekend = (times.day_of_week >= 5).astype(int)
        slot = times.hour * 4 + times.minute // 15
        keys = pd.MultiIndex.from_arrays([np.asarray(detectors), np.asarray(weekend), np.asarray(slot)])
        # np.array(..., copy) because pandas may hand back a read-only view (copy-on-write).
        values = np.array(self.table.reindex(keys).to_numpy(dtype=float), dtype=float, copy=True)
        missing = np.isnan(values)
        if missing.any():
            fallback = self.detector_mean.reindex(np.asarray(detectors)[missing]).to_numpy(dtype=float)
            values[missing] = np.nan_to_num(fallback, nan=0.0)
        return values

    def predict_windows(self, windows) -> np.ndarray:
        return self.predict_for(windows.detectors, windows.times)
