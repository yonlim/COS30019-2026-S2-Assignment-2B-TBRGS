"""Preparing the data and training a model: shared by train.py, evaluate.py and the GUI.

Owner: Lim Yon (105811100)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .data import SeriesScaler, load_traffic_data, make_windows, split_dates
from .models import model_class


@dataclass
class Prepared:
    data: object
    split: object
    scaler: SeriesScaler
    scaled: object
    windows: dict  # "train" / "validation" / "test" -> WindowSet
    lags: int


def load_data(config):
    return load_traffic_data(config.path_of("data", "scats_file"),
                             config.path_of("data", "cache_dir"),
                             config.path_of("data", "site_listing_file"))


def prepare(config, lags: int | None = None, data=None) -> Prepared:
    """Load the data, split it by date, fit the scaler on the training part and build windows."""
    data = data or load_data(config)
    lags = int(lags or config.getint("features", "lags"))
    split = split_dates(data.flows, config.get("data", "train_end"), config.get("data", "validation_end"))
    scaler = SeriesScaler().fit(data.flows, split.train[1])
    scaled = scaler.transform(data.flows)
    windows = {part: make_windows(scaled, lags, *split.part(part))
               for part in ("train", "validation", "test")}
    return Prepared(data, split, scaler, scaled, windows, lags)


def model_params(config, name: str, overrides: dict | None = None) -> dict:
    params = config.section(name)
    params.update({k: v for k, v in (overrides or {}).items() if v not in (None, "")})
    return params


def train_model(config, name: str, prepared: Prepared | None = None, overrides: dict | None = None,
                sample: float | None = None, log=print):
    """Train one model, save it (with the shared scaler) and return (model, prepared).

    sample (0-1] trains on a random share of the training windows, for quick checks.
    """
    params = model_params(config, name, overrides)
    lags = int(params.pop("lags", 0) or config.getint("features", "lags"))
    if prepared is None or prepared.lags != lags:
        prepared = prepare(config, lags, prepared.data if prepared else None)
    train = prepared.windows["train"]
    if sample and 0 < sample < 1:
        rng = np.random.default_rng(0)
        train = train.subset(rng.random(len(train)) < sample)
    log(f"Training {name.upper()} on {len(train):,} windows "
        f"(validation {len(prepared.windows['validation']):,}), {lags} lags")
    model = model_class(name)(lags, params)
    model.fit(train, prepared.windows["validation"], log=log)
    model_dir = config.path_of("paths", "model_dir")
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)
    prepared.scaler.save(os.path.join(model_dir, "scaler.json"))
    log(f"Saved {name} to {model_dir} ({model.meta['train_seconds']:.0f} s)")
    return model, prepared
