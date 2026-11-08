"""XGBoost (gradient-boosted regression trees) traffic flow model: the team's third algorithm.

Owner: Lau Ka Meng (105814507)

Why this model: the unit's example uses a single regression tree. XGBoost
(Chen and Guestrin, 2016) builds hundreds of small regression trees one after
another, each fitted to the errors left by the trees before it, so it keeps
the tree's ability to learn sharp, non-linear rules ("if the last two
intervals were above X at 8 am on a weekday...") while averaging away the
noise of a single tree. It is also a completely different family from the
two recurrent networks, which makes the comparison more informative: it
reads the same information as a flat row of numbers instead of a sequence.

Inputs (see data.flat_features): the 12 lag flows and the target interval's
time of day (sin, cos) and weekend flag. Training stops when the validation
error has not improved for `early_stopping_rounds` trees.
"""

from __future__ import annotations

import numpy as np

from ..data import flat_feature_names, flat_features
from .base import FlowModel

DEFAULTS = {"n_estimators": 2000, "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "min_child_weight": 5, "early_stopping_rounds": 50, "seed": 42}


class XGBoostModel(FlowModel):
    name = "xgboost"
    weights_extension = ".ubj"

    def __init__(self, lags: int, params: dict | None = None):
        merged = dict(DEFAULTS)
        merged.update({k: v for k, v in (params or {}).items() if k in DEFAULTS})
        for key in ("n_estimators", "max_depth", "early_stopping_rounds", "seed"):
            merged[key] = int(merged[key])
        for key in ("learning_rate", "subsample", "colsample_bytree", "min_child_weight"):
            merged[key] = float(merged[key])
        super().__init__(lags, merged)
        self.booster = None

    def _regressor(self):
        from xgboost import XGBRegressor
        p = self.params
        return XGBRegressor(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"], learning_rate=p["learning_rate"],
            subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
            min_child_weight=p["min_child_weight"], early_stopping_rounds=p["early_stopping_rounds"],
            objective="reg:squarederror", tree_method="hist", random_state=p["seed"], n_jobs=-1)

    def _fit(self, train, validation, log) -> dict:
        regressor = self._regressor()
        log(f"  fitting up to {self.params['n_estimators']} trees (early stopping after "
            f"{self.params['early_stopping_rounds']} rounds without improvement)...")
        regressor.fit(train.flat(), train.y, eval_set=[(validation.flat(), validation.y)], verbose=False)
        self.booster = regressor.get_booster()
        val_curve = regressor.evals_result().get("validation_0", {}).get("rmse", [])
        best = int(regressor.best_iteration) + 1
        log(f"  best number of trees: {best}, validation RMSE (scaled) {min(val_curve):.5f}")
        gain = self.booster.get_score(importance_type="gain")
        names = flat_feature_names(self.lags)
        importance = {names[int(key[1:])]: float(value) for key, value in gain.items()}
        total = sum(importance.values()) or 1.0
        return {
            "trees": best,
            "best_val_rmse": float(min(val_curve)) if val_curve else None,
            "feature_importance_gain": {k: round(v / total, 4) for k, v in
                                        sorted(importance.items(), key=lambda kv: -kv[1])},
        }

    def _predict(self, X, times) -> np.ndarray:
        import xgboost
        features = xgboost.DMatrix(flat_features(X, times))
        # iteration_range limits prediction to the trees kept by early stopping.
        best = self.meta.get("trees") if self.meta else None
        if best:
            return self.booster.predict(features, iteration_range=(0, int(best)))
        return self.booster.predict(features)

    def _save_weights(self, path: str) -> None:
        self.booster.save_model(path)

    def _load_weights(self, path: str) -> None:
        import xgboost
        self.booster = xgboost.Booster()
        self.booster.load_model(path)
