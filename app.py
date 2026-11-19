from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "G2.csv"
MODEL_PATH = BASE_DIR / "pokemon_type_model.joblib"
FEATURE_COLUMNS = [
    "hp",
    "attack",
    "defense",
    "height_dm",
    "weight_hg",
    "base_experience",
]
FIELD_LIMITS = {
    "hp": (1, 255),
    "attack": (5, 190),
    "defense": (5, 250),
    "height_dm": (1, 1000),
    "weight_hg": (1, 10000),
    "base_experience": (36, 608),
}
FIELD_LABELS = {
    "hp": "HP",
    "attack": "Attack",
    "defense": "Defense",
    "height_dm": "Height",
    "weight_hg": "Weight",
    "base_experience": "Base Experience",
}

app = Flask(__name__)


def train_model():
    """Train the model from the dataset using the fields accepted by the form."""
    data = pd.read_csv(DATA_PATH)
    missing_columns = set(FEATURE_COLUMNS + ["type_1"]) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    features = data[FEATURE_COLUMNS].copy()
    features.loc[features["weight_hg"].eq(0), "weight_hg"] = np.nan
    target = data["type_1"]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(features, target)
    joblib.dump(model, MODEL_PATH)
    return model


@lru_cache(maxsize=1)
def get_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_model()


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict():
    return render_template("predict.html", values={}, errors={})


@app.route("/result", methods=["POST"])
def result():
    values = request.form.to_dict()
    errors = {}
    features = {}

    for field in FEATURE_COLUMNS:
        raw_value = values.get(field, "").strip()
        if field == "base_experience" and not raw_value:
            features[field] = np.nan
            continue

        try:
            value = float(raw_value)
        except ValueError:
            errors[field] = f"Enter a valid number for {FIELD_LABELS[field]}."
            continue

        minimum, maximum = FIELD_LIMITS[field]
        if not np.isfinite(value) or not minimum <= value <= maximum:
            errors[field] = (
                f"{FIELD_LABELS[field]} must be between {minimum:g} and {maximum:g}."
            )
            continue
        features[field] = value

    if errors:
        return render_template("predict.html", values=values, errors=errors), 400

    input_data = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    predicted_type = get_model().predict(input_data)[0]
    return render_template(
        "result.html",
        predicted_type=predicted_type,
        values=values,
        field_labels=FIELD_LABELS,
    )


if __name__ == "__main__":
    app.run(debug=True)
