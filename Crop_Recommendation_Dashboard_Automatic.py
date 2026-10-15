from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "crop_model.joblib"
REPORT_PATH = ROOT / "reports" / "evaluation.json"
DATA_PATH = ROOT / "data" / "crop_data.csv"
DEFAULT_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
LABEL_COLUMNS = ["label", "crop", "Crop", "Label"]

st.set_page_config(page_title="AI Crop Recommendation Dashboard", page_icon="🌱", layout="wide")

st.markdown("""
<style>
.stApp {
    background: #f6f4e8;
    color: #243528;
}

[data-testid="stSidebar"] {
    background: #e5eddc;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Headings and normal text */
h1, h2, h3, h4, p, label,
.stMarkdown, [data-testid="stCaptionContainer"] {
    color: #243528 !important;
}

/* Main hero banner */
.hero {
    background: linear-gradient(120deg, #1f4b2b, #4e7b43);
    padding: 25px 28px;
    border-radius: 18px;
    margin-bottom: 18px;
    color: white;
}

.hero h1 {
    color: white !important;
    margin: 0 0 7px 0;
}

.hero p {
    color: #edf5e8 !important;
    margin: 0;
}

/* Cards */
.section-card {
    background: white;
    border: 1px solid #dce6d4;
    padding: 17px;
    border-radius: 15px;
}

/* Metric cards and values */
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dce6d4;
    padding: 13px 15px;
    border-radius: 14px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #385441 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #1d482d !important;
    font-weight: 700 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #243528 !important;
}

.stTabs [aria-selected="true"] {
    color: #e84444 !important;
}

/* Dropdown text */
div[data-baseweb="select"] {
    color: #243528 !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
  <h1>🌱 AI Crop Recommendation System</h1>
  <p>Dataset-driven sample conditions • Machine Learning crop suggestions • No API connection</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.info("Ensure models/crop_model.joblib exists. Run train_model.py if needed.")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info("Ensure data/crop_data.csv exists in your project.")
    st.stop()

try:
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    features = model_bundle.get("features", DEFAULT_FEATURES)
    dataset = pd.read_csv(DATA_PATH)
except Exception as exc:
    st.error(f"Could not load the model or dataset: {exc}")
    st.stop()

missing = [feature for feature in features if feature not in dataset.columns]
if missing:
    st.error(f"Dataset is missing required model features: {missing}")
    st.write("Dataset columns:", list(dataset.columns))
    st.stop()

label_col = next((c for c in LABEL_COLUMNS if c in dataset.columns), None)
report = {}
if REPORT_PATH.exists():
    try:
        with REPORT_PATH.open("r", encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, json.JSONDecodeError):
        pass

# Choose a real dataset row per crop where possible. No numeric input fields.
for feature in features:
    dataset[feature] = pd.to_numeric(dataset[feature], errors="coerce")
valid = dataset.dropna(subset=features).reset_index(drop=True)
if valid.empty:
    st.error("No valid numeric samples were found in the dataset.")
    st.stop()

if label_col:
    samples = valid.drop_duplicates(subset=[label_col]).reset_index(drop=True)
else:
    samples = valid.copy().reset_index(drop=True)

accuracy = report.get("accuracy", report.get("test_accuracy"))
accuracy_pct = None
try:
    if accuracy is not None:
        accuracy = float(accuracy)
        accuracy_pct = accuracy if accuracy > 1 else accuracy * 100
except (ValueError, TypeError):
    accuracy_pct = None

summary = st.columns(4)
summary[0].metric("Model Accuracy", f"{accuracy_pct:.2f}%" if accuracy_pct is not None else "Not available")
summary[1].metric("Dataset Rows", f"{len(dataset):,}")
summary[2].metric("Crop Categories", str(dataset[label_col].nunique()) if label_col else "Not labelled")
summary[3].metric("Data Source", "Existing dataset")

crop_tab, weather_tab, accuracy_tab, yield_tab, irrigation_tab = st.tabs([
    "🌱 Crop Recommendation",
    "🌦️ Weather & Conditions",
    "📊 Model Accuracy & Charts",
    "🌾 Crop Yield Prediction",
    "💧 Irrigation Recommendation",
])

with crop_tab:
    st.subheader("AI Crop Recommendation")
    st.write("Select a sample from the existing dataset. All numeric conditions are filled automatically.")
    options = list(samples.index)
    def sample_name(index):
        if label_col:
            return f"Sample for {str(samples.loc[index, label_col]).replace('_', ' ').title()}"
        return f"Dataset sample {index + 1}"
    selected = st.selectbox("Dataset sample", options=options, format_func=sample_name)
    chosen = samples.loc[selected]
    row = pd.DataFrame([{feature: float(chosen[feature]) for feature in features}], columns=features)

    values = st.columns(4)
    values[0].metric("Nitrogen (N)", f"{float(chosen['N']):g}")
    values[1].metric("Phosphorus (P)", f"{float(chosen['P']):g}")
    values[2].metric("Potassium (K)", f"{float(chosen['K']):g}")
    values[3].metric("Soil pH", f"{float(chosen['ph']):.1f}")
    weather = st.columns(3)
    weather[0].metric("Temperature", f"{float(chosen['temperature']):.1f} °C")
    weather[1].metric("Humidity", f"{float(chosen['humidity']):.1f}%")
    weather[2].metric("Rainfall", f"{float(chosen['rainfall']):.1f} mm")

    if st.button("🌾 Recommend Crop", type="primary"):
        try:
            prediction = str(model.predict(row)[0])
            st.session_state["predicted_crop"] = prediction
            st.session_state["predicted_sample"] = int(selected)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(row)[0]
                classifier = model.named_steps.get("classifier", model) if hasattr(model, "named_steps") else model
                classes = getattr(classifier, "classes_", getattr(model, "classes_", []))
                top = pd.DataFrame({"Crop": list(classes), "Probability": probs})
                st.session_state["top_predictions"] = top.sort_values("Probability", ascending=False).head(3)
            else:
                st.session_state["top_predictions"] = None
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

  
    if "predicted_crop" in st.session_state:
        st.success("Model recommendation for the selected dataset sample")
        st.markdown(
            f"# 🌾 {st.session_state['predicted_crop'].replace('_', ' ').title()}"
        )

        top = st.session_state.get("top_predictions")

        if top is not None and not top.empty:
            st.caption(
                "Top 3 model outputs; scores do not guarantee real-world crop success."
            )

            fig = px.bar(
                top.sort_values("Probability"),
                x="Probability",
                y="Crop",
                orientation="h",
                range_x=[0, 1],
                labels={
                    "Probability": "Model probability",
                    "Crop": "Crop"
                },
                color="Probability",
                color_continuous_scale=["#dcebd4", "#2f6d3a"]
            )

            fig.update_layout(
                paper_bgcolor="#f6f4e8",
                plot_bgcolor="#ffffff",
                height=350,
                font=dict(color="#243528", size=14),
                xaxis=dict(
                    title="Model probability",
                    title_font=dict(color="#243528", size=14),
                    tickfont=dict(color="#243528", size=12),
                    gridcolor="#dce6d4",
                    zerolinecolor="#dce6d4"
                ),
                yaxis=dict(
                    title="Crop",
                    title_font=dict(color="#243528", size=14),
                    tickfont=dict(color="#243528", size=13),
                    automargin=True
                ),
                margin=dict(l=100, r=40, t=30, b=60)
            )

            st.plotly_chart(fig, use_container_width=True)

    st.caption("These are dataset examples, not live field measurements.")


with weather_tab:
    st.subheader("Weather & Soil Conditions")
    st.info("No API or live weather connection is used. These charts show values from the selected dataset sample.")
    weather_df = pd.DataFrame({
        "Condition": ["Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "Soil pH"],
        "Value": [float(chosen["temperature"]), float(chosen["humidity"]), float(chosen["rainfall"]), float(chosen["ph"])],
    })
    fig = px.bar(weather_df, x="Condition", y="Value", color="Condition", title="Selected Sample Conditions",
                 color_discrete_sequence=["#4f7b43", "#8eaa69", "#c4d6a4", "#71936a"])
    fig.update_layout(paper_bgcolor="#f6f4e8", plot_bgcolor="white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with accuracy_tab:
    st.subheader("Model Accuracy & Charts")
    if accuracy_pct is not None:
        st.metric("Test Accuracy", f"{accuracy_pct:.2f}%")
        st.progress(max(0.0, min(1.0, accuracy_pct / 100)))
    else:
        st.warning("Accuracy was not found in reports/evaluation.json.")
    st.markdown("""
    <div class="section-card"><b>About accuracy</b><br>
    Test accuracy measures how often the model predicted the correct crop on held-out test examples.
    It does not guarantee success in real farm conditions.</div>
    """, unsafe_allow_html=True)
    if report:
        with st.expander("View evaluation report"):
            st.json(report)
    if label_col:
        counts = dataset[label_col].astype(str).value_counts().rename_axis("Crop").reset_index(name="Samples")
        fig = px.bar(counts, x="Crop", y="Samples", title="Dataset Samples per Crop",
                     color="Samples", color_continuous_scale=["#dcebd4", "#2f6d3a"])
        fig.update_layout(paper_bgcolor="#f6f4e8", plot_bgcolor="white", xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No crop label column found, so crop distribution cannot be displayed.")

with yield_tab:
    st.subheader("Crop Yield Prediction")
    st.warning("Yield prediction is not available yet: the current model predicts crop categories, not yield quantity.")
    st.markdown("""
    A real yield prediction needs a separate dataset with a yield target (such as tonnes per hectare)
    and a separately trained regression model. This dashboard does not invent a yield result.
    """)
    st.markdown('<div class="section-card"><b>Status:</b> Awaiting yield dataset and regression model.</div>', unsafe_allow_html=True)

with irrigation_tab:
    st.subheader("Irrigation Recommendation")
    st.info("The dataset does not provide verified soil-moisture sensor readings. This section is educational guidance only.")
    st.metric("Rainfall in selected sample", f"{float(chosen['rainfall']):.1f} mm")
    st.markdown("""
    - Consider crop type and growth stage.
    - Check soil type and recent local rainfall.
    - Do not decide irrigation needs from one dataset value alone.

    No real-time weather or soil sensor is connected.
    """)

st.divider()
st.caption("AI Crop Recommendation Dashboard • Dataset-based sample values • No manual numeric entry • No external API")
