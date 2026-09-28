"""
predictor.py
------------
Handles training the Random Forest Classifier and making predictions
for new CI/CD pipeline executions.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.data_generator import generate_training_data, get_feature_columns

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH      = os.path.join(BASE_DIR, "model", "random_forest.pkl")
TRAINING_CSV    = os.path.join(BASE_DIR, "data", "training_data.csv")

# ── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Risk thresholds ──────────────────────────────────────────────────────────
RISK_LOW_MAX    = 40   # 0–39  → LOW
RISK_MED_MAX    = 70   # 40–69 → MEDIUM
                       # 70+   → HIGH


def classify_risk(failure_probability: float) -> str:
    """Convert a 0–100 failure probability into a risk label."""
    if failure_probability < RISK_LOW_MAX:
        return "LOW"
    elif failure_probability < RISK_MED_MAX:
        return "MEDIUM"
    else:
        return "HIGH"


def train_model(n_samples: int = 1000, force_retrain: bool = False) -> RandomForestClassifier:
    """
    Train (or load) the Random Forest model.

    If a saved model already exists and force_retrain is False, loads it
    from disk instead of re-training.

    Returns the trained sklearn estimator.
    """
    # Return cached model if available
    if os.path.exists(MODEL_PATH) and not force_retrain:
        log.info("Loading existing model from %s", MODEL_PATH)
        return joblib.load(MODEL_PATH)

    log.info("Training new Random Forest model on %d samples …", n_samples)

    # Generate / reload training data
    df = generate_training_data(n_samples)

    # Persist training data for reference / future reuse
    os.makedirs(os.path.dirname(TRAINING_CSV), exist_ok=True)
    df.to_csv(TRAINING_CSV, index=False)
    log.info("Training data saved to %s", TRAINING_CSV)

    features = get_feature_columns()
    X = df[features]
    y = df["build_success"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    log.info("Model accuracy on test split: %.2f%%", acc * 100)
    log.info("\n%s", classification_report(y_test, y_pred, labels=[0, 1], target_names=["FAILURE", "SUCCESS"], zero_division=0))

    # Persist model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    log.info("Model saved to %s", MODEL_PATH)

    return model


def predict_build(model: RandomForestClassifier, build_features: dict) -> dict:
    """
    Run inference on a single build's features.

    Parameters
    ----------
    model          : trained RandomForestClassifier
    build_features : dict containing all 8 feature values

    Returns a dict with:
      - prediction          : "SUCCESS" or "FAILURE"
      - failure_probability : float 0–100 (rounded to 2 dp)
      - risk_level          : "LOW" | "MEDIUM" | "HIGH"
      - feature_importances : dict of feature → importance score
    """
    features = get_feature_columns()

    # Build a single-row DataFrame so column order is guaranteed
    row = pd.DataFrame([{col: build_features[col] for col in features}])

    # Class order from sklearn: [0=FAILURE, 1=SUCCESS]
    proba = model.predict_proba(row)[0]
    failure_prob_raw = proba[0]          # probability of class 0 (FAILURE)
    failure_probability = round(failure_prob_raw * 100, 2)

    prediction = "SUCCESS" if proba[1] >= proba[0] else "FAILURE"
    risk_level = classify_risk(failure_probability)

    # Feature importances (rounded for readability)
    importances = {
        feat: round(float(imp), 4)
        for feat, imp in zip(features, model.feature_importances_)
    }

    return {
        "prediction":           prediction,
        "failure_probability":  failure_probability,
        "risk_level":           risk_level,
        "feature_importances":  importances,
    }


def get_model_info(model: RandomForestClassifier) -> dict:
    """Return basic metadata about the loaded model for display on the dashboard."""
    return {
        "estimator_type":  type(model).__name__,
        "n_estimators":    model.n_estimators,
        "max_depth":       model.max_depth,
        "n_features":      model.n_features_in_,
        "model_path":      MODEL_PATH,
        "training_csv":    TRAINING_CSV,
    }
