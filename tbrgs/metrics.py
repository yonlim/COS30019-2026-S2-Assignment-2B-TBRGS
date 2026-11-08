"""Error measures used to compare the traffic flow models.

Owner: Lim Yon (105811100)

All functions take actual and predicted flows in vehicles per 15 minutes.

    MAE    mean absolute error, in vehicles per 15 minutes
    RMSE   root mean squared error; punishes large misses (e.g. at peaks) more
    NRMSE  RMSE divided by the mean actual flow (as in the unit's regression
           tree example), so detectors and models can be compared on one scale
    MAPE   mean absolute percentage error, counted only where the actual flow
           is at least MAPE_MIN_FLOW: at night many counts are 0-5 vehicles, and
           one vehicle of error there would be a 20-100% error
    R2     share of the variance in the actual flows that the model explains
"""

from __future__ import annotations

import numpy as np

MAPE_MIN_FLOW = 10.0


def _arrays(actual, predicted):
    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same length")
    if actual.size == 0:
        raise ValueError("no values to compare")
    return actual, predicted


def mae(actual, predicted) -> float:
    actual, predicted = _arrays(actual, predicted)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual, predicted) -> float:
    actual, predicted = _arrays(actual, predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def nrmse(actual, predicted) -> float:
    actual, _ = _arrays(actual, predicted)
    mean = float(np.mean(actual))
    return rmse(actual, predicted) / mean if mean else float("nan")


def mape(actual, predicted, min_flow: float = MAPE_MIN_FLOW) -> float:
    actual, predicted = _arrays(actual, predicted)
    mask = actual >= min_flow
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / actual[mask]) * 100)


def r2(actual, predicted) -> float:
    actual, predicted = _arrays(actual, predicted)
    total = np.sum((actual - actual.mean()) ** 2)
    return float(1 - np.sum((actual - predicted) ** 2) / total) if total else float("nan")


def all_metrics(actual, predicted) -> dict:
    """Every measure at once, as a dict ready for a results table."""
    return {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "NRMSE": nrmse(actual, predicted),
        "MAPE": mape(actual, predicted),
        "R2": r2(actual, predicted),
        "n": int(np.asarray(actual).size),
    }
