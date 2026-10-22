"""
train_model.py
--------------
Trains a Linear Regression model on the California Housing dataset,
evaluates it (MAE, RMSE, R2) and saves everything the website needs
into artifacts/house_model.joblib

Run:  python train_model.py
(app.py also imports build_bundle() and trains automatically if the file is missing.)
"""

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = BASE_DIR / "artifacts" / "house_model.joblib"
DOLLARS_PER_UNIT = 100000  # the dataset's target is in units of $100,000
TARGET = "MedHouseVal"
# Features with extreme outliers: their input limits use the 0.1%-99.9% range
SKEWED = ["AveRooms", "AveBedrms", "Population", "AveOccup"]


def load_data():
    """Load the California Housing dataset (scikit-learn downloads it once)."""
    try:
        return fetch_california_housing(as_frame=True).frame
    except Exception as error:
        raise RuntimeError(
            "Could not load the California Housing dataset. The first run needs an "
            "internet connection so scikit-learn can download it once. "
            f"Please check your connection and try again. Details: {error}"
        )


def build_bundle():
    """Train and evaluate the model; return everything the website needs."""
    df = load_data()
    features = [c for c in df.columns if c != TARGET]  # exact order used by the model
    X, y = df[features], df[TARGET]

    # 80% training / 20% testing split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline: scale the features, then fit Linear Regression
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])
    model.fit(X_train, y_train)

    # Evaluate on the unseen test set
    y_pred = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }

    # Allowed input ranges (hard limits) and "typical" ranges (soft warnings)
    ranges = {}
    for col in features:
        if col in SKEWED:
            low, high = X[col].quantile(0.001), X[col].quantile(0.999)
        else:
            low, high = X[col].min(), X[col].max()
        ranges[col] = {
            "min": float(np.floor(low * 100) / 100),
            "max": float(np.ceil(high * 100) / 100),
            "p1": float(X[col].quantile(0.01)),
            "p99": float(X[col].quantile(0.99)),
            "median": float(X[col].median()),
        }

    return {
        "model": model,
        "model_name": "Linear Regression",
        "feature_names": features,
        "n_records": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": metrics,
        "ranges": ranges,
        "y_test": y_test.to_numpy(),
        "y_pred": y_pred,
        "y_all": y.to_numpy(),
        "corr": df.corr(),
    }


def save_bundle(bundle):
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACT_PATH)


def main():
    try:
        bundle = build_bundle()
    except RuntimeError as error:
        print("\nERROR:", error)
        sys.exit(1)

    m = bundle["metrics"]
    print("Dataset loaded:", bundle["n_records"], "rows")
    print("Features:", bundle["feature_names"])
    print(f"Train rows: {bundle['n_train']} | Test rows: {bundle['n_test']}")
    print("\n--- Model performance on test data ---")
    print(f"MAE  : {m['mae']:.4f}  (about ${m['mae'] * DOLLARS_PER_UNIT:,.0f})")
    print(f"RMSE : {m['rmse']:.4f}  (about ${m['rmse'] * DOLLARS_PER_UNIT:,.0f})")
    print(f"R2   : {m['r2']:.4f}")
    save_bundle(bundle)
    print(f"\nModel saved to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
