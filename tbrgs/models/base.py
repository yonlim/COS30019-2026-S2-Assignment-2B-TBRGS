"""The interface every flow model follows.

Owner: Lim Yon (105811100), agreed with the whole team.

A model is trained on WindowSet objects (see tbrgs/data.py) and predicts
scaled flows; SeriesScaler turns them back into vehicles per 15 minutes.
Each saved model is two files in the model folder:

    <name>.json        settings, training history and timing (this class)
    <name>.<ext>       the fitted weights (written by the subclass)
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod

import numpy as np


class FlowModel(ABC):
    name = "base"          # short name used in files, the CLI and the GUI
    weights_extension = ""  # extension of the weights file

    def __init__(self, lags: int, params: dict | None = None):
        self.lags = int(lags)
        self.params = dict(params or {})
        self.meta = {}  # filled in by fit(): history, timings, sizes

    # ----- to be written by each model -------------------------------------
    @abstractmethod
    def _fit(self, train, validation, log) -> dict:
        """Train on WindowSets; return a dict describing the training run."""

    @abstractmethod
    def _predict(self, X: np.ndarray, times) -> np.ndarray:
        """Predict scaled flows from X of shape (n, lags, 4) for target interval start times."""

    @abstractmethod
    def _save_weights(self, path: str) -> None: ...

    @abstractmethod
    def _load_weights(self, path: str) -> None: ...

    # ----- shared behaviour ---------------------------------------------------
    def fit(self, train, validation, log=print) -> dict:
        if train.lags != self.lags:
            raise ValueError(f"windows have {train.lags} lags but the model expects {self.lags}")
        started = time.perf_counter()
        details = self._fit(train, validation, log) or {}
        self.meta = {
            "model": self.name,
            "lags": self.lags,
            "params": self.params,
            "train_windows": len(train),
            "validation_windows": len(validation),
            "train_seconds": round(time.perf_counter() - started, 2),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **details,
        }
        return self.meta

    def predict(self, X: np.ndarray, times) -> np.ndarray:
        """Predict scaled flows for windows X (n, lags, 4) whose targets start at times (n,)."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3 or X.shape[1] != self.lags:
            raise ValueError(f"expected X of shape (n, {self.lags}, 4), got {X.shape}")
        if len(X) != len(times):
            raise ValueError("X and times must have the same length")
        if len(X) == 0:
            return np.zeros(0, dtype=float)
        # Flows cannot be negative.
        return np.clip(np.asarray(self._predict(X, times), dtype=float).ravel(), 0.0, None)

    def predict_windows(self, windows) -> np.ndarray:
        return self.predict(windows.X, windows.times)

    def save(self, model_dir: str) -> None:
        os.makedirs(model_dir, exist_ok=True)
        self._save_weights(self.weights_path(model_dir))
        with open(os.path.join(model_dir, f"{self.name}.json"), "w", encoding="utf-8") as file:
            json.dump(self.meta or {"model": self.name, "lags": self.lags, "params": self.params},
                      file, indent=2, default=float)

    @classmethod
    def load(cls, model_dir: str) -> "FlowModel":
        meta_path = os.path.join(model_dir, f"{cls.name}.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No trained {cls.name} model in '{model_dir}'. Train it first: "
                                    f"python train.py --model {cls.name}")
        with open(meta_path, encoding="utf-8") as file:
            meta = json.load(file)
        model = cls(meta["lags"], meta.get("params"))
        model.meta = meta
        model._load_weights(model.weights_path(model_dir))
        return model

    def weights_path(self, model_dir: str) -> str:
        return os.path.join(model_dir, f"{self.name}{self.weights_extension}")
