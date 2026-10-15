"""
Train and evaluate a crop recommendation classifier.
Expected CSV columns:
N, P, K, temperature, humidity, ph, rainfall, label
Run: python train_model.py
"""
from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "crop_data.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place a crop recommendation CSV at data/crop_data.csv. "
            "Expected columns: N, P, K, temperature, humidity, ph, rainfall, label."
        )

    df = pd.read_csv(DATA_PATH)
    # Normalize common column-name variants from public crop datasets.
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}
    rename = {}
    for expected in FEATURES + [TARGET]:
        if expected not in df.columns and expected.lower() in lower_map:
            rename[lower_map[expected.lower()]] = expected
    df = df.rename(columns=rename)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df = df[FEATURES + [TARGET]].copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET] = df[TARGET].astype("string").str.strip()
    df = df.dropna(subset=[TARGET])
    df = df.dropna(subset=FEATURES)
    df = df[df[TARGET] != ""]
    if len(df) < 20:
        raise ValueError("Dataset has too few complete rows. Please use a larger dataset.")

    X, y = df[FEATURES], df[TARGET].astype(str)
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced"
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump({"model": model, "features": FEATURES, "target": TARGET},
                MODEL_DIR / "crop_model.joblib")
    report = {
        "rows_used": int(len(df)),
        "number_of_crops": int(y.nunique()),
        "features": FEATURES,
        "model": "RandomForestClassifier",
        "test_size": 0.2,
        "random_state": 42,
        "accuracy": round(float(accuracy), 4),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=sorted(y.unique())
        ).tolist(),
        "class_labels": sorted(y.unique()),
    }
    with open(REPORT_DIR / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Rows used: {len(df)}")
    print(f"Crop classes: {y.nunique()}")
    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Saved model: {MODEL_DIR / 'crop_model.joblib'}")
    print(f"Saved report: {REPORT_DIR / 'evaluation.json'}")

if __name__ == "__main__":
    main()
