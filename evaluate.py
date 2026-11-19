"""Evaluate and compare the traffic flow models on the test week (25-31 Oct 2006).

Owners: Lim Yon (model accuracy, sections 1-3) and Lau Ka Meng (route impact, section 4).

Usage (from this folder):
    python evaluate.py              every trained model + the two baselines, with figures
    python evaluate.py --no-figures

Writes to results/:
    metrics.csv              overall accuracy, training time and prediction speed per model
    metrics_by_period.csv    accuracy in the morning peak, evening peak, daytime, night, weekend
    metrics_by_hour.csv      mean absolute error for each hour of the day
    metrics_by_site.csv      mean absolute error per SCATS site
    pairwise_tests.csv       is model A really better than model B? (block bootstrap)
    route_impact.csv         how prediction errors change the routes the system recommends
    model_comparison.png, error_by_hour.png, example_day.png, training_curves.png

1. Accuracy: every model predicts each detector's next 15-minute count from
   its previous 12 counts, for every complete window in the test week.
2. Breakdowns by time of day, period and site show where models differ.
3. Pairwise tests: errors of neighbouring 15-minute intervals are strongly
   related, so a plain t-test would overstate the evidence. Instead the
   detector-days are resampled (block bootstrap, 100 times) to get a 95%
   interval for the difference in mean absolute error between two models.
4. Route impact: for sampled origin-destination pairs and times, the fastest
   route is computed from each model's predicted flows and from the observed
   flows. "ETA error" is how far the predicted travel time is from the time
   the same route takes with the observed flows; "regret" is how much slower
   the recommended route is than the truly fastest route.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from tbrgs.config import load_config
from tbrgs.metrics import all_metrics, mae
from tbrgs.models import DISPLAY_NAMES, load_model, trained_models
from tbrgs.models.baselines import HistoricalAverageModel, PersistenceModel
from tbrgs.training import prepare

PERIODS = {
    "AM peak (weekday 07:00-09:00)": lambda t: (t.day_of_week < 5) & (t.hour >= 7) & (t.hour < 9),
    "PM peak (weekday 16:00-19:00)": lambda t: (t.day_of_week < 5) & (t.hour >= 16) & (t.hour < 19),
    "Weekday daytime (09:00-16:00)": lambda t: (t.day_of_week < 5) & (t.hour >= 9) & (t.hour < 16),
    "Night (00:00-06:00)": lambda t: t.hour < 6,
    "Weekend (all day)": lambda t: t.day_of_week >= 5,
}


def predict_all(config, prepared, log=print) -> dict:
    """Predictions (vehicles per 15 min) of every source on the test windows, plus timing."""
    test = prepared.windows["test"]
    scaler = prepared.scaler
    sources = {"persistence": PersistenceModel(),
               "historical": HistoricalAverageModel().fit(prepared.scaled, prepared.split.train[1])}
    model_dir = config.path_of("paths", "model_dir")
    for name in trained_models(model_dir):
        model = load_model(name, model_dir)
        if model.lags != prepared.lags:
            log(f"Skipping {name}: trained with {model.lags} lags, config.ini says {prepared.lags}")
            continue
        sources[name] = model
    outputs = {}
    for name, source in sources.items():
        started = time.perf_counter()
        scaled = source.predict_windows(test)
        seconds = time.perf_counter() - started
        outputs[name] = {
            "predicted": scaler.inverse(np.clip(scaled, 0, None), test.detectors),
            "ms_per_1000": seconds / len(test) * 1e6,
            "meta": getattr(source, "meta", {}),
        }
        log(f"  {DISPLAY_NAMES.get(name, name):<20} predicted {len(test):,} windows in {seconds:.2f} s")
    return outputs


def block_bootstrap(errors_a, errors_b, blocks, repeats=100, seed=0) -> tuple:
    """95% interval for mean(|e_a|) - mean(|e_b|), resampling whole blocks (detector-days)."""
    frame = pd.DataFrame({"d": np.abs(errors_a) - np.abs(errors_b), "block": blocks})
    per_block = frame.groupby("block")["d"].agg(["sum", "count"])
    sums, counts = per_block["sum"].to_numpy(), per_block["count"].to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(repeats, len(sums)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return float(sums.sum() / counts.sum()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def accuracy_tables(prepared, outputs) -> dict:
    test = prepared.windows["test"]
    actual = prepared.scaler.inverse(test.y, test.detectors)
    times = pd.DatetimeIndex(test.times)
    sites = prepared.data.detectors.loc[test.detectors, "site"].to_numpy()
    blocks = pd.Series(test.detectors).astype(str).to_numpy() + "|" + times.strftime("%Y-%m-%d")

    overall, by_period, by_hour, by_site = [], [], [], []
    for name, out in outputs.items():
        predicted = out["predicted"]
        row = {"model": DISPLAY_NAMES.get(name, name), "key": name, **all_metrics(actual, predicted),
               "ms_per_1000": out["ms_per_1000"]}
        meta = out["meta"] or {}
        row["train_seconds"] = meta.get("train_seconds")
        row["epochs_or_trees"] = meta.get("epochs_run", meta.get("trees"))
        row["parameters"] = meta.get("parameters")
        overall.append(row)
        for period, rule in PERIODS.items():
            mask = np.asarray(rule(times))
            by_period.append({"model": row["model"], "period": period,
                              **all_metrics(actual[mask], predicted[mask])})
        hours = times.hour
        for hour in range(24):
            mask = np.asarray(hours == hour)
            by_hour.append({"model": row["model"], "hour": hour, "MAE": mae(actual[mask], predicted[mask])})
        for site in np.unique(sites):
            mask = sites == site
            by_site.append({"model": row["model"], "site": int(site), "MAE": mae(actual[mask], predicted[mask]),
                            "mean_flow": float(actual[mask].mean())})

    names = list(outputs)
    pairwise = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff, low, high = block_bootstrap(actual - outputs[a]["predicted"], actual - outputs[b]["predicted"],
                                              blocks)
            verdict = ("A better" if high < 0 else "B better" if low > 0 else "no clear difference")
            pairwise.append({"model_A": DISPLAY_NAMES.get(a, a), "model_B": DISPLAY_NAMES.get(b, b),
                             "MAE_A_minus_B": diff, "ci95_low": low, "ci95_high": high, "verdict": verdict})
    return {
        "metrics": pd.DataFrame(overall).sort_values("MAE"),
        "metrics_by_period": pd.DataFrame(by_period),
        "metrics_by_hour": pd.DataFrame(by_hour),
        "metrics_by_site": pd.DataFrame(by_site),
        "pairwise_tests": pd.DataFrame(pairwise),
    }


def route_impact(config, sources, pairs=60, seed=1, log=print) -> pd.DataFrame:
    """Section 4: does a better flow model give better route recommendations? (Lau Ka Meng)

    For `pairs` random origin-destination pairs and every hour from 06:00 to
    21:00 on each test day, the fastest route is found with each source's
    predicted flows and with the observed flows (the "truth").
    """
    from tbrgs.routing import build_search_problem, top_k_routes
    from tbrgs.system import TBRGS

    system = TBRGS(config)
    travel = system.travel_time_model()
    network = system.network()
    positions = {s: network.position(s) for s in network.sites.index}
    rng = np.random.default_rng(seed)
    sites = list(network.sites.index)
    od = set()
    while len(od) < pairs:
        a, b = rng.choice(sites, 2, replace=False)
        od.add((int(a), int(b)))
    od = sorted(od)
    test_start = system.split.test[0]
    times = [test_start + pd.Timedelta(days=d, hours=h) for d in range(7) for h in range(6, 22)]

    rows = []
    for when in times:
        truth_est, _ = system.edge_estimates("observed", when, travel)
        truth_cost = {key: e.time_s for key, e in truth_est.items()}
        nodes, truth_edges = build_search_problem(positions, truth_cost, travel.free_flow_seconds_per_km())
        best_truth = {}
        for o, d in od:
            found, _ = top_k_routes(nodes, truth_edges, o, d, 1, "AS")
            best_truth[(o, d)] = found[0][0]
        for source in sources:
            est, _ = system.edge_estimates(source, when, travel)
            cost = {key: e.time_s for key, e in est.items()}
            _, edges = build_search_problem(positions, cost, travel.free_flow_seconds_per_km())
            for o, d in od:
                found, _ = top_k_routes(nodes, edges, o, d, 1, "AS")
                predicted_time, path = found[0]
                actual_time = sum(truth_cost[(a, b)] for a, b in zip(path, path[1:]))
                rows.append({"source": DISPLAY_NAMES.get(source, source), "time": when, "origin": o,
                             "destination": d, "predicted_s": predicted_time, "actual_s": actual_time,
                             "best_s": best_truth[(o, d)],
                             "eta_error_s": predicted_time - actual_time,
                             "regret_s": actual_time - best_truth[(o, d)]})
    log(f"  route impact: {len(od)} origin-destination pairs x {len(times)} times x {len(sources)} sources")
    return pd.DataFrame(rows)


def make_figures(results_dir, tables, prepared, outputs, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = tables["metrics"]
    order = list(metrics["model"])
    colours = {"LSTM": "#1565c0", "GRU": "#7b1fa2", "XGBoost": "#2e7d32",
               "Historical average": "#9e9e9e", "Persistence": "#616161"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for axis, column, title in zip(axes, ("MAE", "RMSE", "MAPE"),
                                   ("MAE (vehicles / 15 min)", "RMSE (vehicles / 15 min)", "MAPE (%)")):
        values = metrics[column].to_numpy()
        bars = axis.bar(order, values, color=[colours.get(m, "#888") for m in order])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=30)
        axis.set_ylim(min(values) * 0.85, max(values) * 1.05)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Test week (25-31 Oct 2006): one-step (15-minute) prediction error, all 140 detectors")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "model_comparison.png"), dpi=110)
    plt.close(fig)

    hourly = tables["metrics_by_hour"]
    fig, axis = plt.subplots(figsize=(9, 3.8))
    for model in order:
        part = hourly[hourly["model"] == model]
        axis.plot(part["hour"], part["MAE"], marker="o", markersize=3, label=model, color=colours.get(model))
    axis.set_xlabel("Hour of day")
    axis.set_ylabel("MAE (vehicles / 15 min)")
    axis.set_title("Prediction error by hour of day, test week")
    axis.set_xticks(range(0, 24, 2))
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "error_by_hour.png"), dpi=110)
    plt.close(fig)

    # One busy detector on one test weekday.
    test = prepared.windows["test"]
    actual = prepared.scaler.inverse(test.y, test.detectors)
    times = pd.DatetimeIndex(test.times)
    day = pd.Timestamp("2006-10-26")
    busiest = pd.Series(actual).groupby(test.detectors).mean().idxmax()
    mask = (test.detectors == busiest) & (times.normalize() == day)
    fig, axis = plt.subplots(figsize=(9, 3.8))
    axis.plot(times[mask], actual[mask], color="black", linewidth=2, label="Observed")
    for name, out in outputs.items():
        label = DISPLAY_NAMES.get(name, name)
        axis.plot(times[mask], out["predicted"][mask], linewidth=1, label=label, color=colours.get(label))
    location = prepared.data.detectors.loc[busiest, "location"]
    axis.set_title(f"Detector {busiest} ({location}), {day:%a %d %b %Y}")
    axis.set_ylabel("Vehicles per 15 minutes")
    import matplotlib.dates as mdates
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "example_day.png"), dpi=110)
    plt.close(fig)

    curves = {name: out["meta"].get("history") for name, out in outputs.items()
              if out["meta"] and out["meta"].get("history")}
    if curves:
        fig, axis = plt.subplots(figsize=(7, 3.6))
        for name, history in curves.items():
            label = DISPLAY_NAMES.get(name, name)
            axis.plot(range(1, len(history["loss"]) + 1), history["loss"], linestyle="--",
                      color=colours.get(label), label=f"{label} training")
            axis.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"],
                      color=colours.get(label), label=f"{label} validation")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean squared error (scaled)")
        axis.set_title("LSTM and GRU learning curves")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "training_curves.png"), dpi=110)
        plt.close(fig)


def run(config=None, figures=True, routes=True, log=print) -> dict:
    config = config or load_config()
    results_dir = config.path_of("paths", "results_dir")
    os.makedirs(results_dir, exist_ok=True)
    log("Preparing the test windows...")
    prepared = prepare(config)
    log(f"Test week: {len(prepared.windows['test']):,} windows from {prepared.data.flows.shape[1]} detectors")
    outputs = predict_all(config, prepared, log)
    tables = accuracy_tables(prepared, outputs)
    if routes:
        impact = route_impact(config, [n for n in outputs if n != "persistence"], log=log)
        tables["route_impact"] = impact
        summary = impact.groupby("source").agg(
            mean_abs_eta_error_s=("eta_error_s", lambda s: s.abs().mean()),
            mean_eta_bias_s=("eta_error_s", "mean"),
            mean_regret_s=("regret_s", "mean"),
            share_fastest_route=("regret_s", lambda s: (s < 0.5).mean()),
            worst_regret_s=("regret_s", "max")).reset_index()
        tables["route_impact_summary"] = summary
    for name, table in tables.items():
        table.to_csv(os.path.join(results_dir, f"{name}.csv"), index=False)
    if figures:
        make_figures(results_dir, tables, prepared, outputs, config)
    columns = ["model", "MAE", "RMSE", "NRMSE", "MAPE", "R2", "train_seconds", "ms_per_1000"]
    log("\n" + tables["metrics"][columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if "route_impact_summary" in tables:
        log("\n" + tables["route_impact_summary"].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    log("\n" + tables["pairwise_tests"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log(f"\nResults written to {results_dir}")
    return tables


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the TBRGS flow models on the test week.")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-routes", action="store_true", help="skip the route-impact study")
    args = parser.parse_args(argv)
    run(figures=not args.no_figures, routes=not args.no_routes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
