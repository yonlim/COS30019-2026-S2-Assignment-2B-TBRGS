"""Train the traffic flow models and save them in saved_models/.

Usage (from this folder):
    python train.py --model all              LSTM, GRU and XGBoost with config.ini settings
    python train.py --model lstm             one model
    python train.py --model gru --epochs 10  override a setting
    python train.py --model all --quick      2 epochs on 10% of the data (checks the code runs)

Training uses the targets of 1-21 Oct 2006, stops early on 22-24 Oct, and
never sees the test week (25-31 Oct). The models are saved with the scaler
(saved_models/scaler.json) so the route finder can use them straight away.
"""

import argparse
import sys

from tbrgs.config import load_config
from tbrgs.data import SeriesScaler
from tbrgs.metrics import all_metrics
from tbrgs.models import MODEL_CLASSES
from tbrgs.training import train_model


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the TBRGS traffic flow models.")
    parser.add_argument("--model", default="all", choices=[*MODEL_CLASSES, "all"])
    parser.add_argument("--epochs", type=int, help="epochs for LSTM/GRU (overrides config.ini)")
    parser.add_argument("--quick", action="store_true", help="2 epochs / 200 trees on 10%% of the data")
    parser.add_argument("--config", help="another configuration file")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    names = list(MODEL_CLASSES) if args.model == "all" else [args.model]
    overrides = {}
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.quick:
        overrides.update({"epochs": 2, "n_estimators": 200})

    prepared = None
    for name in names:
        model, prepared = train_model(config, name, prepared, overrides,
                                      sample=0.1 if args.quick else None)
        test = prepared.windows["test"]
        scaler: SeriesScaler = prepared.scaler
        actual = scaler.inverse(test.y, test.detectors)
        predicted = scaler.inverse(model.predict_windows(test), test.detectors)
        scores = all_metrics(actual, predicted)
        print(f"{name.upper()} on the test week: MAE {scores['MAE']:.2f}  RMSE {scores['RMSE']:.2f}  "
              f"MAPE {scores['MAPE']:.1f}%  R2 {scores['R2']:.3f}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
