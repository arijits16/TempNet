"""
Step 3: Build and train the ANN (7 -> 16 -> 8 -> 2).

Run from the project root (after preprocess):
    python -m src.train
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow info logs

import pandas as pd
from tensorflow import keras

from src import HISTORY_PATH, MODEL_PATH, MODELS_DIR, RESULTS_DIR, SEED
from src.preprocess import load_splits

EPOCHS = 100
BATCH_SIZE = 16
PATIENCE = 10


def build_model(n_inputs: int = 7, n_outputs: int = 2) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=(n_inputs,), name="weather_today"),
        keras.layers.Dense(16, activation="relu", name="hidden_1"),
        keras.layers.Dense(8, activation="relu", name="hidden_2"),
        keras.layers.Dense(n_outputs, activation="linear", name="temps_tomorrow"),
    ], name="TempNet")
    model.compile(optimizer=keras.optimizers.Adam(), loss="mse", metrics=["mae"])
    return model


def main() -> None:
    keras.utils.set_random_seed(SEED)  # seeds Python, NumPy and TensorFlow
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    splits, _, _ = load_splits()
    model = build_model()
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1,
    )
    history = model.fit(
        splits["train"]["X"], splits["train"]["y"],
        validation_data=(splits["val"]["X"], splits["val"]["y"]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,  # shuffles mini-batches inside the training set only
        callbacks=[early_stop],
        verbose=2,
    )

    model.save(MODEL_PATH)
    hist = pd.DataFrame(history.history)
    hist.index = hist.index + 1
    hist.to_csv(HISTORY_PATH, index_label="epoch")

    best_epoch = int(hist["val_loss"].idxmin())
    print(f"\nTrained for {len(hist)} epochs; best epoch = {best_epoch} "
          f"(val MSE, scaled = {hist['val_loss'].min():.5f})")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved training history to {HISTORY_PATH}")


if __name__ == "__main__":
    main()
