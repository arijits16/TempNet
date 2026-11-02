"""
Step 4: Evaluate the trained ANN on the held-out test set.

- Predictions are converted back to degrees Celsius before scoring.
- The ANN is compared against a persistence baseline
  ("tomorrow's temperature = today's temperature").

Run from the project root (after train):
    python -m src.evaluate
"""

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras

from src import HISTORY_PATH, METRICS_PATH, MODEL_PATH, PREDICTIONS_PATH, RESULTS_DIR
from src.preprocess import load_splits


def scores(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 4),
    }


def plot_loss(path) -> None:
    hist = pd.read_csv(HISTORY_PATH)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hist["epoch"], hist["loss"], label="Training loss", color="#2563eb", linewidth=2)
    ax.plot(hist["epoch"], hist["val_loss"], label="Validation loss", color="#ea580c", linewidth=2)
    best = hist.loc[hist["val_loss"].idxmin()]
    ax.axvline(best["epoch"], color="#6b7280", linestyle="--", linewidth=1,
               label=f"Best epoch ({int(best['epoch'])})")
    ax.set_title("Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (scaled values, log scale)")
    ax.set_yscale("log")  # early epochs are much larger; log scale shows later progress
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_actual_vs_predicted(dates, actual, ann, baseline, label, path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, actual, label="Actual", color="#111827", linewidth=2)
    ax.plot(dates, ann, label="ANN prediction", color="#2563eb", linewidth=1.8)
    ax.plot(dates, baseline, label="Persistence baseline", color="#9ca3af",
            linewidth=1.2, linestyle="--")
    ax.set_title(f"Actual vs Predicted Next-Day {label} Temperature (Test Set)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    splits, _, y_scaler = load_splits()
    model = keras.models.load_model(MODEL_PATH)

    test = splits["test"]
    frame = test["frame"]

    # ANN predictions, converted from scaled values back to deg C
    ann = y_scaler.inverse_transform(model.predict(test["X"], verbose=0))
    actual = frame[["next_max_temperature", "next_min_temperature"]].to_numpy()
    # Persistence baseline: tomorrow = today
    baseline = frame[["max_temperature", "min_temperature"]].to_numpy()

    metrics = {"test_samples": len(frame)}
    for i, name in enumerate(["max_temperature", "min_temperature"]):
        metrics[name] = {
            "ann": scores(actual[:, i], ann[:, i]),
            "baseline": scores(actual[:, i], baseline[:, i]),
        }
    metrics["invalid_ann_predictions_max_le_min"] = int((ann[:, 0] <= ann[:, 1]).sum())

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    predictions = pd.DataFrame({
        # The prediction is FOR the next day, so report the target date
        "input_date": frame["date"],
        "target_date": (pd.to_datetime(frame["date"]) + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"),
        "actual_max": actual[:, 0],
        "ann_pred_max": np.round(ann[:, 0], 2),
        "baseline_pred_max": baseline[:, 0],
        "actual_min": actual[:, 1],
        "ann_pred_min": np.round(ann[:, 1], 2),
        "baseline_pred_min": baseline[:, 1],
    })
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    target_dates = pd.to_datetime(predictions["target_date"])
    plot_loss(RESULTS_DIR / "loss_curve.png")
    plot_actual_vs_predicted(target_dates, actual[:, 0], ann[:, 0], baseline[:, 0],
                             "Maximum", RESULTS_DIR / "max_temperature.png")
    plot_actual_vs_predicted(target_dates, actual[:, 1], ann[:, 1], baseline[:, 1],
                             "Minimum", RESULTS_DIR / "min_temperature.png")

    print(f"Test samples: {len(frame)}")
    print(f"{'Target':<17}{'Model':<10}{'MAE (°C)':>10}{'RMSE (°C)':>11}")
    for name in ["max_temperature", "min_temperature"]:
        for model_name in ["ann", "baseline"]:
            m = metrics[name][model_name]
            print(f"{name:<17}{model_name:<10}{m['mae']:>10.3f}{m['rmse']:>11.3f}")
    print(f"\nSaved metrics, predictions and graphs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
