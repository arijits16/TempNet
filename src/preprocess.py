"""
Step 2: Prepare the synthetic CSV for ANN training.

- Inputs  : the 7 weather values of day t
- Targets : max and min temperature of day t+1 (made by shifting one row up)
- Split   : 70% train / 15% validation / 15% test, in date order (no shuffling)
- Scaling : MinMax scalers fitted on the TRAINING rows only (no data leakage)

Run from the project root:
    python -m src.preprocess
"""

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src import (
    FEATURES, INPUT_SCALER_PATH, MODELS_DIR, PROCESSED_CSV, PROCESSED_DIR,
    RAW_CSV, TARGET_SCALER_PATH, TARGETS,
)

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15


def build_supervised(df: pd.DataFrame) -> pd.DataFrame:
    """Pair each day's conditions with the NEXT day's temperatures."""
    df = df.sort_values("date").reset_index(drop=True)
    out = df[["date"] + FEATURES].copy()
    out["next_max_temperature"] = df["max_temperature"].shift(-1)
    out["next_min_temperature"] = df["min_temperature"].shift(-1)
    # The last day has no "tomorrow" in the data, so it cannot be a training sample
    return out.iloc[:-1].reset_index(drop=True)


def assign_splits(n: int) -> list[str]:
    train_end = int(n * TRAIN_FRACTION)
    val_end = int(n * (TRAIN_FRACTION + VAL_FRACTION))
    return ["train"] * train_end + ["val"] * (val_end - train_end) + ["test"] * (n - val_end)


def load_splits():
    """Load the processed CSV and saved scalers; return scaled arrays per split.

    Used by train.py and evaluate.py so both see exactly the same data.
    """
    data = pd.read_csv(PROCESSED_CSV)
    x_scaler = joblib.load(INPUT_SCALER_PATH)
    y_scaler = joblib.load(TARGET_SCALER_PATH)

    splits = {}
    for name in ["train", "val", "test"]:
        part = data[data["split"] == name].reset_index(drop=True)
        splits[name] = {
            "frame": part,
            "X": x_scaler.transform(part[FEATURES]),
            "y": y_scaler.transform(part[TARGETS]),
        }
    return splits, x_scaler, y_scaler


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW_CSV)
    data = build_supervised(raw)
    data["split"] = assign_splits(len(data))

    train = data[data["split"] == "train"]
    x_scaler = MinMaxScaler().fit(train[FEATURES])
    y_scaler = MinMaxScaler().fit(train[TARGETS])

    data.to_csv(PROCESSED_CSV, index=False)
    joblib.dump(x_scaler, INPUT_SCALER_PATH)
    joblib.dump(y_scaler, TARGET_SCALER_PATH)

    print(f"Supervised samples: {len(data)} (last day dropped: it has no next-day target)")
    for name in ["train", "val", "test"]:
        part = data[data["split"] == name]
        print(f"  {name:<5}: {len(part):>3} rows  {part['date'].iloc[0]} -> {part['date'].iloc[-1]}")
    print(f"Saved processed data to {PROCESSED_CSV}")
    print(f"Saved scalers (fitted on training rows only) to {MODELS_DIR}")


if __name__ == "__main__":
    main()
