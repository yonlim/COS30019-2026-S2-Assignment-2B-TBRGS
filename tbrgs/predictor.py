"""Turning a trained model into flow predictions for the whole network at one time.

Owner: Lim Yon (105811100)

For a chosen time t, every detector's last `lags` measured intervals
(t - lags ... t - 1) are fed to the model, which predicts the flow in the
interval starting at t (one step, 15 minutes, ahead). This is how the system
would run live: it always knows the counts up to now and predicts the next
interval.

Times outside the dataset (it covers 1-31 October 2006 only) are mapped to
the same weekday and time in the last week of the data, and the result says
so. A detector whose recent history is incomplete (a missing day, or the
first three hours of 1 October) falls back to the historical average for its
time of day, and the number of such detectors is reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import INTERVAL, N_STEP_FEATURES, make_windows, time_features

# Prediction sources that are not trained models.
OBSERVED = "observed"      # the measured flow itself (for comparison only)
HISTORICAL = "historical"  # the historical-average baseline


class FlowPredictor:
    def __init__(self, data, scaler, historical):
        self.data = data
        self.scaler = scaler
        self.historical = historical
        self.scaled = scaler.transform(data.flows)
        self._values = self.scaled.to_numpy(dtype=np.float32)
        self._time_index = {t: i for i, t in enumerate(self.scaled.index)}
        self._cache = {}

    # ----- time handling ------------------------------------------------------
    def resolve_time(self, when) -> tuple:
        """Round down to a 15-minute boundary and map it into the dataset if needed.

        Returns (time used, note) where note is "" if the time was inside the data.
        """
        requested = pd.Timestamp(when).floor("15min")
        first, last = self.scaled.index[0], self.scaled.index[-1]
        if first <= requested <= last:
            return requested, ""
        last_week_start = (last - pd.Timedelta(days=6)).normalize()
        offset = (requested.day_of_week - last_week_start.day_of_week) % 7
        used = last_week_start + pd.Timedelta(days=offset) + (requested - requested.normalize())
        note = (f"{requested:%Y-%m-%d %H:%M} is outside the data (1-31 Oct 2006); "
                f"using the same weekday and time, {used:%a %d %b %Y %H:%M}.")
        return used, note

    # ----- predictions ------------------------------------------------------------
    def predict(self, model, when) -> tuple:
        """Predict every detector's flow (vehicles per 15 minutes) for the interval starting at `when`.

        model is a trained FlowModel, or the strings OBSERVED / HISTORICAL.
        Returns (pd.Series detector -> flow, info dict).
        """
        used, note = self.resolve_time(when)
        name = model if isinstance(model, str) else model.name
        key = (name, used)
        if key in self._cache:
            flows, info = self._cache[key]
            return flows.copy(), dict(info, note=note, requested=str(pd.Timestamp(when)))

        detectors = list(self.scaled.columns)
        index = self._time_index[used]
        fallback = self.historical.predict_for(detectors, [used] * len(detectors))
        info = {"time_used": str(used), "model": name, "fallback_detectors": 0}

        if name == OBSERVED:
            scaled = self._values[index].astype(float)
            missing = np.isnan(scaled)
            scaled[missing] = fallback[missing]
            info["fallback_detectors"] = int(missing.sum())
        elif name == HISTORICAL:
            scaled = fallback
        else:
            lags = model.lags
            scaled = fallback.copy()
            if index >= lags:
                window = self._values[index - lags:index]            # (lags, S)
                complete = ~np.isnan(window).any(axis=0)
                if complete.any():
                    steps = self.scaled.index[index - lags:index]
                    X = np.empty((int(complete.sum()), lags, N_STEP_FEATURES), dtype=np.float32)
                    X[:, :, 0] = window[:, complete].T
                    X[:, :, 1:] = time_features(steps)[None, :, :]
                    scaled[complete] = model.predict(X, [used] * len(X))
                info["fallback_detectors"] = int((~complete).sum())
            else:
                info["fallback_detectors"] = len(detectors)
        flows = pd.Series(self.scaler.inverse(np.clip(scaled, 0, None), detectors), index=detectors)
        self._cache[key] = (flows, info)
        return flows.copy(), dict(info, note=note, requested=str(pd.Timestamp(when)))

    def day_series(self, model, detector: str, day) -> pd.DataFrame:
        """Observed and one-step-ahead predicted flow for one detector over one day.

        Used by the GUI's prediction chart. All 96 intervals are predicted in
        one batch.
        """
        day, _ = self.resolve_time(pd.Timestamp(day).normalize())
        day = day.normalize()
        times = pd.date_range(day, day + pd.Timedelta(days=1) - INTERVAL, freq=INTERVAL)
        observed = self.data.flows.loc[times, detector].to_numpy(dtype=float)
        name = model if isinstance(model, str) else model.name
        scaled = self.historical.predict_for([detector] * len(times), times)
        if name == OBSERVED:
            return pd.DataFrame({"observed": observed, "predicted": observed}, index=times)
        if name != HISTORICAL:
            windows = make_windows(self.scaled[[detector]], model.lags, times[0], times[-1])
            if len(windows):
                positions = times.get_indexer(pd.DatetimeIndex(windows.times))
                scaled[positions] = model.predict(windows.X, windows.times)
        predicted = self.scaler.inverse(np.clip(scaled, 0, None), [detector] * len(times))
        return pd.DataFrame({"observed": observed, "predicted": predicted}, index=times)
