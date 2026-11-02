"""
TempNet: Streamlit dashboard for ANN-based next-day temperature prediction.

Run from the project root (after generate_dataset, preprocess, train, evaluate):
    streamlit run app.py
"""

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import pandas as pd
import streamlit as st

from src import (
    FEATURES, INPUT_SCALER_PATH, METRICS_PATH, MODEL_PATH, PREDICTIONS_PATH,
    PROCESSED_CSV, RAW_CSV, RESULTS_DIR, TARGET_SCALER_PATH,
)

st.set_page_config(page_title="TempNet - Next-Day Temperature ANN", page_icon="🌡️", layout="wide")

SYNTHETIC_NOTE = (
    "**Synthetic data notice:** every record in this project was generated locally with NumPy. "
    "It is demonstration data, **not real weather observations**, and predictions are **not a real weather forecast**."
)

# label, unit, min allowed, max allowed, step
INPUT_SPECS = {
    "max_temperature": ("Maximum temperature", "°C", -10.0, 50.0, 0.1),
    "min_temperature": ("Minimum temperature", "°C", -15.0, 45.0, 0.1),
    "humidity": ("Humidity", "%", 0.0, 100.0, 0.5),
    "wind_speed": ("Wind speed", "km/h", 0.0, 100.0, 0.5),
    "rainfall": ("Rainfall", "mm", 0.0, 300.0, 0.5),
    "cloud_cover": ("Cloud cover", "%", 0.0, 100.0, 0.5),
    "solar_radiation": ("Solar radiation", "W/m²", 0.0, 400.0, 1.0),
}


# ---------- Cached loaders: the model is loaded once, never retrained ----------

@st.cache_resource(show_spinner="Loading trained ANN...")
def load_model_and_scalers():
    from tensorflow import keras  # imported lazily so other sections open fast
    model = keras.models.load_model(MODEL_PATH)
    return model, joblib.load(INPUT_SCALER_PATH), joblib.load(TARGET_SCALER_PATH)


@st.cache_data
def load_raw():
    df = pd.read_csv(RAW_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_training_ranges():
    data = pd.read_csv(PROCESSED_CSV)
    return data[data["split"] == "train"][FEATURES].agg(["min", "max"])


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


def missing_artifacts() -> list[str]:
    needed = [RAW_CSV, PROCESSED_CSV, MODEL_PATH, INPUT_SCALER_PATH, TARGET_SCALER_PATH,
              METRICS_PATH, PREDICTIONS_PATH]
    return [str(p.relative_to(RAW_CSV.parent.parent)) for p in needed if not p.exists()]


# ---------- Sections ----------

def section_overview():
    st.header("A. Project Overview")
    st.markdown(
        "**Objective:** design an Artificial Neural Network (ANN) that uses **today's** weather "
        "conditions to predict **tomorrow's maximum and minimum temperatures**."
    )
    st.info(SYNTHETIC_NOTE)

    df = load_raw()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Input features", len(FEATURES))
    c4.metric("Outputs", 2)

    st.subheader("What is an ANN?")
    st.markdown(
        "An Artificial Neural Network is a model loosely inspired by the brain. It is made of layers of "
        "simple units called **neurons**. Each neuron multiplies its inputs by **weights**, adds a **bias**, "
        "and passes the result through an **activation function**. During training, the network compares its "
        "predictions with the true answers and adjusts the weights a little at a time "
        "(**backpropagation + gradient descent**) so that the error gets smaller."
    )

    st.subheader("ANN architecture")
    left, right = st.columns([3, 2])
    with left:
        st.graphviz_chart(
            """
            digraph G {
              rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica"];
              in  [label="Input layer\\n7 features\\n(today)", fillcolor="#dbeafe"];
              h1  [label="Hidden layer 1\\n16 neurons\\nReLU", fillcolor="#e0e7ff"];
              h2  [label="Hidden layer 2\\n8 neurons\\nReLU", fillcolor="#e0e7ff"];
              out [label="Output layer\\n2 neurons\\nLinear", fillcolor="#dcfce7"];
              in -> h1 -> h2 -> out;
            }
            """
        )
    with right:
        st.table(pd.DataFrame({
            "Layer": ["Input", "Hidden 1", "Hidden 2", "Output"],
            "Neurons": [7, 16, 8, 2],
            "Activation": ["Not Applicable", "ReLU", "ReLU", "Linear"],
            "Parameters": [0, 7 * 16 + 16, 16 * 8 + 8, 8 * 2 + 2],
        }).set_index("Layer"))
        st.caption("Total trainable parameters: 128 + 136 + 18 = 282")

    st.markdown(
        "**Inputs (day t):** max temperature, min temperature, humidity, wind speed, rainfall, "
        "cloud cover, solar radiation.  \n**Outputs (day t+1):** max temperature, min temperature.  \n"
        "**Training:** Adam optimizer, MSE loss, batch size 16, up to 100 epochs with early stopping."
    )


def section_prediction():
    st.header("B. Next-Day Prediction")
    st.warning("Prediction made with the ANN trained on **synthetic data**. This is **not a real weather forecast**.")

    df = load_raw()
    defaults = df[FEATURES].iloc[-1]  # pre-fill with the last day in the dataset
    train_range = load_training_ranges()

    with st.form("predict_form"):
        cols = st.columns(2)
        values = {}
        for i, feat in enumerate(FEATURES):
            label, unit, lo, hi, step = INPUT_SPECS[feat]
            values[feat] = cols[i % 2].number_input(
                f"{label} ({unit})", min_value=lo, max_value=hi,
                value=float(defaults[feat]), step=step, format="%.1f",
                help=f"Allowed range {lo:g} to {hi:g} {unit}",
            )
        submitted = st.form_submit_button("Predict Tomorrow's Temperature", type="primary")

    if not submitted:
        st.caption("Fields are pre-filled with the last day of the synthetic dataset. Adjust them and click the button.")
        return

    if values["max_temperature"] <= values["min_temperature"]:
        st.error("Today's maximum temperature must be greater than today's minimum temperature. Please correct the inputs.")
        return

    outside = [
        INPUT_SPECS[f][0] for f in FEATURES
        if not train_range.loc["min", f] <= values[f] <= train_range.loc["max", f]
    ]
    if outside:
        st.info(
            "These inputs are outside the range seen during training, so the prediction is an extrapolation "
            f"and may be less reliable: {', '.join(outside)}."
        )

    model, x_scaler, y_scaler = load_model_and_scalers()
    x = pd.DataFrame([values], columns=FEATURES)
    pred = y_scaler.inverse_transform(model.predict(x_scaler.transform(x), verbose=0))[0]
    pred_max, pred_min = float(pred[0]), float(pred[1])

    st.subheader("Predicted temperatures for tomorrow")
    c1, c2 = st.columns(2)
    c1.metric("Maximum temperature", f"{pred_max:.2f} °C",
              delta=f"{pred_max - values['max_temperature']:+.2f} °C vs today")
    c2.metric("Minimum temperature", f"{pred_min:.2f} °C",
              delta=f"{pred_min - values['min_temperature']:+.2f} °C vs today")

    if pred_max <= pred_min:
        st.warning(
            f"The model predicted a maximum ({pred_max:.2f} °C) that is not above the minimum "
            f"({pred_min:.2f} °C). The raw prediction is shown unchanged; this usually means the inputs are unusual."
        )

    st.subheader("Selected input conditions (today)")
    st.table(pd.DataFrame({
        "Parameter": [INPUT_SPECS[f][0] for f in FEATURES],
        "Value": [f"{values[f]:.1f} {INPUT_SPECS[f][1]}" for f in FEATURES],
    }).set_index("Parameter"))


def section_performance():
    st.header("C. Model Performance")
    metrics = load_metrics()
    st.markdown(
        f"Evaluated on the **held-out test set** ({metrics['test_samples']} days at the end of the timeline, "
        "never seen during training). All errors are in **°C**."
    )

    rows = []
    for target, label in [("max_temperature", "Max temperature"), ("min_temperature", "Min temperature")]:
        ann, base = metrics[target]["ann"], metrics[target]["baseline"]
        rows.append({
            "Target": label,
            "ANN MAE": ann["mae"], "Baseline MAE": base["mae"],
            "ANN RMSE": ann["rmse"], "Baseline RMSE": base["rmse"],
            "MAE improvement": f"{(1 - ann['mae'] / base['mae']) * 100:+.1f}%",
        })
    table = pd.DataFrame(rows).set_index("Target")
    st.dataframe(table.style.format(precision=3), width="stretch")

    c1, c2 = st.columns(2)
    for col, target, label in [(c1, "max_temperature", "Max"), (c2, "min_temperature", "Min")]:
        ann, base = metrics[target]["ann"], metrics[target]["baseline"]
        col.metric(f"{label} temp - ANN MAE", f"{ann['mae']:.3f} °C",
                   delta=f"{ann['mae'] - base['mae']:+.3f} °C vs baseline", delta_color="inverse")

    st.caption(
        "**Persistence baseline:** predicts that tomorrow's temperature equals today's. "
        "A useful model should have lower error than this simple rule. A positive improvement means the ANN is better."
    )

    st.subheader("Error metrics explained")
    st.markdown(
        "- **MSE (Mean Squared Error):** average of the squared errors. Large errors are penalised heavily. "
        "Used as the training loss.\n"
        "- **MAE (Mean Absolute Error):** average size of the error in °C. *\"On average the prediction is off by X °C.\"*\n"
        "- **RMSE (Root Mean Squared Error):** square root of MSE, back in °C. It is always ≥ MAE; "
        "a big gap between them means a few large mistakes."
    )

    st.subheader("Graphs")
    st.image(str(RESULTS_DIR / "loss_curve.png"), caption="Training vs validation loss (scaled MSE, log scale)")
    st.image(str(RESULTS_DIR / "max_temperature.png"), caption="Actual vs predicted next-day maximum temperature")
    st.image(str(RESULTS_DIR / "min_temperature.png"), caption="Actual vs predicted next-day minimum temperature")

    with st.expander("Test-set predictions table"):
        st.dataframe(pd.read_csv(PREDICTIONS_PATH), width="stretch")


def section_dataset():
    st.header("D. Dataset Explorer")
    st.info(SYNTHETIC_NOTE)
    df = load_raw()

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Date range", f"{df['date'].min():%d %b %Y} → {df['date'].max():%d %b %Y}")
    st.markdown("**Columns:** " + ", ".join(f"`{c}`" for c in df.columns))

    n = st.slider("Rows to preview", 5, 50, 10)
    st.dataframe(df.head(n).assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")),
                 width="stretch", hide_index=True)

    st.subheader("Temperature over time")
    st.line_chart(df.set_index("date")[["max_temperature", "min_temperature"]], y_label="°C",
                  color=["#ea580c", "#2563eb"])

    st.subheader("Other weather parameters")
    param = st.selectbox(
        "Parameter", ["humidity", "wind_speed", "rainfall", "cloud_cover", "solar_radiation"],
        format_func=lambda f: f"{INPUT_SPECS[f][0]} ({INPUT_SPECS[f][1]})",
    )
    if param == "rainfall":
        st.bar_chart(df.set_index("date")[[param]], y_label="mm")
    else:
        st.line_chart(df.set_index("date")[[param]], y_label=INPUT_SPECS[param][1])

    with st.expander("Summary statistics"):
        st.dataframe(df.drop(columns=["date", "day"]).describe().T.round(2), width="stretch")


def section_methodology():
    st.header("E. ANN Methodology")
    steps = [
        ("1. Synthetic CSV generation", "`src/generate_dataset.py` creates 500 consecutive days of weather with "
         "seasonal sine curves, day-to-day persistence, cloud-driven rain and solar radiation, and random noise "
         "(fixed seed 42). Saved to `data/weather_data.csv`."),
        ("2. Data preprocessing", "`src/preprocess.py` pairs each day's 7 conditions with the next day's max/min "
         "temperature (shift by one row, last row dropped), splits 70/15/15 **in date order**, and fits MinMax "
         "scalers on the **training rows only** to avoid leakage."),
        ("3. ANN training", "`src/train.py` trains a 7-16-8-2 network with Adam and MSE, batch size 16, up to 100 "
         "epochs. Early stopping watches validation loss and restores the best weights."),
        ("4. Next-day prediction", "The saved model and scalers are loaded; inputs are scaled, passed through the "
         "network, and outputs are converted back to °C."),
        ("5. Evaluation", "`src/evaluate.py` computes MAE and RMSE on the test set, compares with the persistence "
         "baseline, and saves metrics, predictions and graphs to `results/`."),
    ]
    st.markdown("**Synthetic CSV generation → Data preprocessing → ANN training → Next-day prediction → Evaluation**")
    for title, text in steps:
        with st.container(border=True):
            st.markdown(f"**{title}**  \n{text}")


# ---------- Layout ----------

st.title("🌡️ TempNet: Next-Day Temperature Prediction with an ANN")
st.caption("Soft Computing mini-project · runs fully offline · synthetic demonstration data")

missing = missing_artifacts()
if missing:
    st.error(
        "Some project files are missing. Run the pipeline first from the project root:\n\n"
        "```\npython -m src.generate_dataset\npython -m src.preprocess\npython -m src.train\npython -m src.evaluate\n```\n\n"
        "Missing: " + ", ".join(missing)
    )
    st.stop()

tabs = st.tabs(["Overview", "Prediction", "Performance", "Dataset", "Methodology"])
with tabs[0]:
    section_overview()
with tabs[1]:
    section_prediction()
with tabs[2]:
    section_performance()
with tabs[3]:
    section_dataset()
with tabs[4]:
    section_methodology()
