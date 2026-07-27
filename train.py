"""
train.py
--------
Trains a simple Linear Regression model to predict Salary based on
YearsExperience, EducationLevel and Age. Saves the trained model and
the encoder used for EducationLevel using joblib so they can be
loaded later by the Flask app (app.py) for deployment.
"""

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "salary_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)

    # 2. Encode categorical column (EducationLevel)
    encoder = LabelEncoder()
    df["EducationLevelEncoded"] = encoder.fit_transform(df["EducationLevel"])

    # 3. Features & target
    X = df[["YearsExperience", "EducationLevelEncoded", "Age"]]
    y = df["Salary"]

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Train model (kept simple: Linear Regression)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("Model Evaluation:")
    print(f"  MAE  : {mae:,.2f}")
    print(f"  RMSE : {rmse:,.2f}")
    print(f"  R^2  : {r2:.4f}")

    # 7. Save model + encoder for use in the Flask app
    joblib.dump(model, os.path.join(MODEL_DIR, "salary_model.pkl"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "education_encoder.pkl"))
    print(f"\nSaved model to {MODEL_DIR}/salary_model.pkl")
    print(f"Saved encoder to {MODEL_DIR}/education_encoder.pkl")


if __name__ == "__main__":
    main()
