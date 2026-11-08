"""Shared code for the two recurrent networks (LSTM and GRU).

Owners: Lim Yon (LSTM) and Jonathan Yu (GRU). Both networks use exactly the
same layers, data and training settings, so any difference in their results
comes from the recurrent cell alone.

Network (Keras 3 / TensorFlow), adapted from the unit's example repository
(hyswin/Example-LSTM-and-GRU-for-A2B-COS30019-) for Keras 3:

    Input (lags, 4)   scaled flow + sin/cos time of day + weekend flag per step
    cell(64, return_sequences=True)
    cell(64)
    Dropout(0.2)
    Dense(1)          linear output, so peaks above the training maximum
                      can still be predicted (the example used a sigmoid)

Training: Adam, mean squared error, batch 256, up to 40 epochs, early stopping
on the validation loss (patience 6) keeping the best weights.
"""

from __future__ import annotations

import time

import numpy as np

from ..data import N_STEP_FEATURES
from .base import FlowModel

DEFAULTS = {"units": 64, "layers": 2, "dropout": 0.2, "epochs": 40, "batch_size": 256,
            "learning_rate": 0.001, "patience": 6, "seed": 42}


def _keras():
    """Import Keras only when a recurrent model is used (TensorFlow is slow to load)."""
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's start-up messages
    import keras
    return keras


class RecurrentFlowModel(FlowModel):
    cell = "LSTM"  # the Keras layer class name; set by the subclass
    weights_extension = ".keras"

    def __init__(self, lags: int, params: dict | None = None):
        merged = dict(DEFAULTS)
        merged.update({k: v for k, v in (params or {}).items() if k in DEFAULTS})
        for key in ("units", "layers", "epochs", "batch_size", "patience", "seed"):
            merged[key] = int(merged[key])
        for key in ("dropout", "learning_rate"):
            merged[key] = float(merged[key])
        super().__init__(lags, merged)
        self.network = None

    def build(self):
        keras = _keras()
        layer_class = getattr(keras.layers, self.cell)
        inputs = keras.Input(shape=(self.lags, N_STEP_FEATURES), name="history")
        x = inputs
        for i in range(self.params["layers"]):
            last = i == self.params["layers"] - 1
            x = layer_class(self.params["units"], return_sequences=not last, name=f"{self.cell.lower()}_{i + 1}")(x)
        x = keras.layers.Dropout(self.params["dropout"], name="dropout")(x)
        outputs = keras.layers.Dense(1, name="next_flow")(x)
        network = keras.Model(inputs, outputs, name=self.name)
        network.compile(optimizer=keras.optimizers.Adam(learning_rate=self.params["learning_rate"]),
                        loss="mse", metrics=["mae"])
        return network

    def _fit(self, train, validation, log) -> dict:
        keras = _keras()
        keras.utils.set_random_seed(self.params["seed"])
        self.network = self.build()

        class EpochLog(keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                self.started = time.perf_counter()

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                log(f"  epoch {epoch + 1:2d}: loss {logs.get('loss', float('nan')):.5f}  "
                    f"val_loss {logs.get('val_loss', float('nan')):.5f}  "
                    f"({time.perf_counter() - self.started:.1f} s)")

        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.params["patience"],
                                             restore_best_weights=True)
        history = self.network.fit(
            train.X, train.y,
            validation_data=(validation.X, validation.y),
            epochs=self.params["epochs"], batch_size=self.params["batch_size"],
            shuffle=True, verbose=0, callbacks=[stop, EpochLog()])
        losses = {key: [float(v) for v in values] for key, values in history.history.items()}
        val_loss = losses.get("val_loss", [])
        return {
            "epochs_run": len(losses.get("loss", [])),
            "best_epoch": int(np.argmin(val_loss)) + 1 if val_loss else None,
            "best_val_loss": float(min(val_loss)) if val_loss else None,
            "parameters": int(self.network.count_params()),
            "history": losses,
        }

    def _predict(self, X, times) -> np.ndarray:
        if len(X) <= 2048:
            # Calling the network directly is much faster than predict() for the
            # small batches the route finder uses (one row per detector).
            return self.network(X, training=False).numpy().ravel()
        return self.network.predict(X, batch_size=4096, verbose=0).ravel()

    def _save_weights(self, path: str) -> None:
        self.network.save(path)

    def _load_weights(self, path: str) -> None:
        self.network = _keras().models.load_model(path)
