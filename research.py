"""Research experiments for the report's Research section.

Usage (from this folder):
    python research.py coordinates   SCATS coordinate correction         (Jonathan Yu)
    python research.py lags          how much history the models need     (Lim Yon)
    python research.py horizon       predicting 15, 30 and 60 minutes ahead (Lim Yon)
    python research.py profile       telling the model each detector's usual day (Lim Yon)
    python research.py methods       the Part A methods on the real road network (Lau Ka Meng)
    python research.py all

Each study writes results/research_<name>.csv (and a figure where useful).
The lags, horizon and profile studies use XGBoost because it trains in
seconds, so many settings can be compared; the conclusions are then checked
against the LSTM/GRU results in the main evaluation.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd

from tbrgs.config import load_config
from tbrgs.data import make_windows
from tbrgs.metrics import all_metrics
from tbrgs.models.baselines import HistoricalAverageModel
from tbrgs.training import prepare


def _results_dir(config):
    path = config.path_of("paths", "results_dir")
    os.makedirs(path, exist_ok=True)
    return path


def _xgboost(config, train, validation):
    from tbrgs.models.xgboost_model import XGBoostModel
    model = XGBoostModel(train.lags, config.section("xgboost"))
    model.fit(train, validation, log=lambda line: None)
    return model


# ---------------------------------------------------------------------------
# Coordinates (Jonathan Yu)
# ---------------------------------------------------------------------------
_SUFFIX = re.compile(r"\b(RD|ST|HWY|GV|AVE|FWY|PDE|CRES|DR)\b")


def _road_key(name: str) -> str:
    return " ".join(_SUFFIX.sub("", str(name).upper().replace("_", " ")).split())


def study_coordinates(config, log=print) -> pd.DataFrame:
    """Estimate the shift between SCATS coordinates and the real intersections.

    data/Traffic_Count_Locations_with_LONG_LAT.csv (VicRoads) lists traffic
    count points with accurate coordinates. For each SCATS site, the nearest
    INTERSECTION count point (within 1 km) whose description joins two of
    the site's roads with '&' (e.g. "BURKE RD & TOORAK RD") is taken as the
    true position. The median difference over all matched sites is the
    shift; the median is used because a few count points sit on one corner
    of a large intersection.
    """
    prepared_data = prepare(config).data
    counts = pd.read_csv(os.path.join(os.path.dirname(config.path_of("data", "scats_file")),
                                      "Traffic_Count_Locations_with_LONG_LAT.csv"),
                         usecols=["X", "Y", "TFM_TYP_DE", "SITE_DESC"])
    counts = counts[(counts["TFM_TYP_DE"] == "INTERSECTION") & counts["SITE_DESC"].str.contains("&", na=False)]
    counts = counts.reset_index(drop=True)
    descriptions = [{_road_key(p) for p in d.upper().split("&")} for d in counts["SITE_DESC"]]
    xs, ys = counts["X"].to_numpy(), counts["Y"].to_numpy()
    rows = []
    detectors = prepared_data.detectors
    for site, info in prepared_data.sites.iterrows():
        roads = {_road_key(r) for r in detectors.loc[detectors["site"] == site, ["road", "cross_road"]].values.ravel()}
        lat, lon = info["latitude"], info["longitude"]
        match = np.array([len(roads & d) >= 2 for d in descriptions])
        if not match.any():
            continue
        distance = np.hypot((ys - lat) * 111.2, (xs - lon) * 111.2 * np.cos(np.radians(lat)))
        distance[~match] = np.inf
        best = int(np.argmin(distance))
        if distance[best] <= 1.0:
            rows.append({"site": site, "name": info["name"], "count_point": counts.at[best, "SITE_DESC"],
                         "distance_m": distance[best] * 1000, "dlat": ys[best] - lat, "dlon": xs[best] - lon})
    table = pd.DataFrame(rows)
    dlat, dlon = table["dlat"].median(), table["dlon"].median()
    after = np.hypot((table["dlat"] - dlat) * 111200,
                     (table["dlon"] - dlon) * 111200 * np.cos(np.radians(-37.82)))
    log(f"Matched {len(table)} of {len(prepared_data.sites)} sites to an intersection count point.")
    log(f"Median shift: latitude {dlat:+.5f} deg ({dlat * 111200:+.0f} m), "
        f"longitude {dlon:+.5f} deg ({dlon * 111200 * np.cos(np.radians(-37.82)):+.0f} m)")
    log(f"Median distance to the count point: {table['distance_m'].median():.0f} m before, "
        f"{np.median(after):.0f} m after the shift.")
    table["distance_after_m"] = after
    table.to_csv(os.path.join(_results_dir(config), "research_coordinates.csv"), index=False)
    return table


# ---------------------------------------------------------------------------
# Lags, horizon and profile (Lim Yon)
# ---------------------------------------------------------------------------
def study_lags(config, log=print, lag_options=(1, 2, 4, 8, 12, 16, 24)) -> pd.DataFrame:
    """Test accuracy of XGBoost with different amounts of history (1 interval = 15 min)."""
    rows = []
    base = prepare(config)
    for lags in lag_options:
        p = prepare(config, lags, base.data)
        model = _xgboost(config, p.windows["train"], p.windows["validation"])
        test = p.windows["test"]
        actual = p.scaler.inverse(test.y, test.detectors)
        scores = all_metrics(actual, p.scaler.inverse(model.predict_windows(test), test.detectors))
        rows.append({"lags": lags, "history_minutes": lags * 15, **scores, "trees": model.meta["trees"]})
        log(f"  lags {lags:2d}: MAE {scores['MAE']:.2f}  RMSE {scores['RMSE']:.2f}  test windows {len(test):,}")
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(_results_dir(config), "research_lags.csv"), index=False)
    return table


def _horizon_windows(scaled, lags, horizon, start, end):
    """Windows whose target is `horizon` intervals after the last input."""
    windows = make_windows(scaled, lags + horizon - 1, start, end)
    windows.X = windows.X[:, :lags]
    return windows


def study_horizon(config, log=print, horizons=(1, 2, 4, 8)) -> pd.DataFrame:
    """Direct multi-step prediction: one XGBoost model per horizon (15, 30, 60, 120 min)."""
    base = prepare(config)
    lags = base.lags
    historical = HistoricalAverageModel().fit(base.scaled, base.split.train[1])
    rows = []
    for horizon in horizons:
        parts = {name: _horizon_windows(base.scaled, lags, horizon, *base.split.part(name))
                 for name in ("train", "validation", "test")}
        model = _xgboost(config, parts["train"], parts["validation"])
        test = parts["test"]
        actual = base.scaler.inverse(test.y, test.detectors)
        for name, scaled in (("XGBoost", model.predict_windows(test)),
                             ("Persistence", test.X[:, -1, 0]),
                             ("Historical average", historical.predict_windows(test))):
            scores = all_metrics(actual, base.scaler.inverse(scaled, test.detectors))
            rows.append({"horizon_minutes": horizon * 15, "model": name, **scores})
        log(f"  {horizon * 15:3d} min ahead: XGBoost MAE {rows[-3]['MAE']:.2f}, persistence "
            f"{rows[-2]['MAE']:.2f}, historical {rows[-1]['MAE']:.2f}")
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(_results_dir(config), "research_horizon.csv"), index=False)
    return table


def study_profile(config, log=print) -> pd.DataFrame:
    """Does knowing each detector's usual flow at this time of day help?

    The models only see the last 3 hours and the time of day, so they cannot
    tell a detector whose peak is in the morning from one whose peak is in
    the evening until the peak starts. This study adds one input: the
    detector's historical average for the target interval (weekday or weekend),
    computed from the training weeks only. In the training data the average
    includes the target itself (one of about 15 values), a small leak that
    can only make the model trust the profile slightly too much; validation
    and test data are not affected.
    """
    from xgboost import XGBRegressor
    base = prepare(config)
    historical = HistoricalAverageModel().fit(base.scaled, base.split.train[1])
    params = config.section("xgboost")
    rows = []
    for variant in ("lags + time (as in the system)", "lags + time + usual flow"):
        sets = {}
        for name in ("train", "validation", "test"):
            w = base.windows[name]
            features = w.flat()
            if "usual" in variant:
                features = np.column_stack([features, historical.predict_windows(w)])
            sets[name] = (features, w.y)
        model = XGBRegressor(n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]),
                             learning_rate=float(params["learning_rate"]), subsample=float(params["subsample"]),
                             colsample_bytree=float(params["colsample_bytree"]),
                             min_child_weight=float(params["min_child_weight"]),
                             early_stopping_rounds=int(params["early_stopping_rounds"]), tree_method="hist",
                             random_state=int(params["seed"]), n_jobs=-1)
        model.fit(*sets["train"], eval_set=[sets["validation"]], verbose=False)
        test = base.windows["test"]
        actual = base.scaler.inverse(test.y, test.detectors)
        predicted = base.scaler.inverse(np.clip(model.predict(sets["test"][0]), 0, None), test.detectors)
        scores = all_metrics(actual, predicted)
        times = pd.DatetimeIndex(test.times)
        peak = np.asarray((times.day_of_week < 5) & (((times.hour >= 7) & (times.hour < 9)) |
                                                   ((times.hour >= 16) & (times.hour < 19))))
        peak_scores = all_metrics(actual[peak], predicted[peak])
        rows.append({"variant": variant, **scores, "peak_MAE": peak_scores["MAE"], "peak_RMSE": peak_scores["RMSE"]})
        log(f"  {variant:<32} MAE {scores['MAE']:.2f}  RMSE {scores['RMSE']:.2f}  peak MAE {peak_scores['MAE']:.2f}")
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(_results_dir(config), "research_profile.csv"), index=False)
    return table


# ---------------------------------------------------------------------------
# Part A methods on the real network (Lau Ka Meng)
# ---------------------------------------------------------------------------
def study_methods(config, log=print, when="2006-10-25 08:00") -> pd.DataFrame:
    """Run all six Part A methods for every origin-destination pair at one time.

    Edge costs come from the best trained model available (LSTM if trained).
    UCS (CUS1) gives the true fastest time, so each method's extra time can
    be measured.
    """
    from tbrgs.routing import METHODS, build_search_problem, top_k_routes
    from tbrgs.system import TBRGS

    system = TBRGS(config)
    sources = system.available_sources()
    source = "lstm" if "lstm" in sources else sources[0]
    travel = system.travel_time_model()
    network = system.network()
    estimates, _ = system.edge_estimates(source, when, travel)
    positions = {s: network.position(s) for s in network.sites.index}
    nodes, edges = build_search_problem(positions, {k: e.time_s for k, e in estimates.items()},
                                        travel.free_flow_seconds_per_km())
    sites = list(network.sites.index)
    pairs = [(a, b) for a in sites for b in sites if a != b]
    optimal = {}
    rows = []
    for method in ["CUS1", "AS", "BFS", "DFS", "GBFS", "CUS2"]:
        started = time.perf_counter()
        for a, b in pairs:
            goal, created, path = METHODS[method](nodes, edges, a, [b])
            cost = sum(c for x, y in zip(path, path[1:]) for n, c in edges[x] if n == y)
            if method == "CUS1":
                optimal[(a, b)] = cost
            rows.append({"method": method, "origin": a, "destination": b, "cost_s": cost,
                         "nodes_created": created, "moves": len(path) - 1})
        elapsed = time.perf_counter() - started
        log(f"  {method:<5} {len(pairs)} searches in {elapsed * 1000:.0f} ms")
    table = pd.DataFrame(rows)
    table["optimal_s"] = [optimal[(a, b)] for a, b in zip(table["origin"], table["destination"])]
    table["extra_percent"] = (table["cost_s"] / table["optimal_s"] - 1) * 100
    summary = table.groupby("method").agg(
        fastest_found_percent=("extra_percent", lambda s: (s < 1e-6).mean() * 100),
        mean_extra_percent=("extra_percent", "mean"),
        worst_extra_percent=("extra_percent", "max"),
        mean_nodes_created=("nodes_created", "mean"),
        mean_moves=("moves", "mean")).reindex(["CUS1", "AS", "GBFS", "BFS", "CUS2", "DFS"]).reset_index()

    # Top-5 routes with A*: how different are the alternatives?
    spreads, overlaps = [], []
    for a, b in pairs[::7]:
        found, _ = top_k_routes(nodes, edges, a, b, 5, "AS")
        if len(found) >= 2:
            spreads.append((found[-1][0] / found[0][0] - 1) * 100)
            first = set(zip(found[0][1], found[0][1][1:]))
            for _, path in found[1:]:
                links = set(zip(path, path[1:]))
                overlaps.append(len(first & links) / max(len(links), 1) * 100)
    summary.attrs["note"] = (f"Flows: {source} at {when}. Top-5 (A*): route 5 is on average "
                             f"{np.mean(spreads):.1f}% slower than route 1; alternatives share "
                             f"{np.mean(overlaps):.0f}% of their links with route 1.")
    log(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    log(summary.attrs["note"])
    results_dir = _results_dir(config)
    table.to_csv(os.path.join(results_dir, "research_methods_detail.csv"), index=False)
    summary.to_csv(os.path.join(results_dir, "research_methods.csv"), index=False)
    with open(os.path.join(results_dir, "research_methods_note.txt"), "w", encoding="utf-8") as file:
        file.write(summary.attrs["note"] + "\n")
    return summary


STUDIES = {"coordinates": study_coordinates, "lags": study_lags, "horizon": study_horizon,
           "profile": study_profile, "methods": study_methods}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="TBRGS research experiments.")
    parser.add_argument("study", choices=[*STUDIES, "all"])
    args = parser.parse_args(argv)
    config = load_config()
    for name in (STUDIES if args.study == "all" else [args.study]):
        print(f"\n== {name} ==")
        STUDIES[name](config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
