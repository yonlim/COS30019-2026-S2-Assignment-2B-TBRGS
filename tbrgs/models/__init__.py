"""Task 2: the traffic flow prediction models.

Every model predicts the flow of the next 15-minute interval for a detector
from that detector's last `lags` intervals (scaled by SeriesScaler). All of
them share the FlowModel interface in base.py, so the evaluation, the GUI
and the route finder can use any of them in the same way.

    lstm.py           Long Short-Term Memory network     (Lim Yon)
    gru.py            Gated Recurrent Unit network       (Jonathan Yu)
    xgboost_model.py  gradient-boosted regression trees  (Lau Ka Meng)
    baselines.py      persistence and historical average, used only as
                      reference points in the evaluation
"""

from __future__ import annotations

import os

from .base import FlowModel

# Name used on the command line / in the GUI -> (module, class name).
MODEL_CLASSES = {
    "lstm": (".lstm", "LSTMModel"),
    "gru": (".gru", "GRUModel"),
    "xgboost": (".xgboost_model", "XGBoostModel"),
}
DISPLAY_NAMES = {"lstm": "LSTM", "gru": "GRU", "xgboost": "XGBoost",
                 "persistence": "Persistence", "historical": "Historical average"}


def model_class(name: str):
    """Return the class for a model name, importing its module only when needed.

    TensorFlow takes several seconds to import, so it is only loaded when an
    LSTM or GRU model is actually used.
    """
    import importlib
    try:
        module_name, class_name = MODEL_CLASSES[name]
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_CLASSES)}") from None
    module = importlib.import_module(module_name, __name__)
    return getattr(module, class_name)


def trained_models(model_dir: str) -> list:
    """Names of the models that have been trained and saved in model_dir."""
    return [name for name in MODEL_CLASSES
            if os.path.exists(os.path.join(model_dir, f"{name}.json"))]


def load_model(name: str, model_dir: str) -> FlowModel:
    return model_class(name).load(model_dir)


__all__ = ["FlowModel", "MODEL_CLASSES", "DISPLAY_NAMES", "model_class", "trained_models", "load_model"]
