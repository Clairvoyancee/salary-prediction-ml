
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "salary_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "education_encoder.pkl")

app = Flask(__name__)

# Load model + encoder once at startup
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
EDUCATION_OPTIONS = list(encoder.classes_)  # e.g. ['Bachelors', 'Masters', 'PhD']


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        education_options=EDUCATION_OPTIONS,
        prediction=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        years_experience = float(request.form["years_experience"])
        education_level = request.form["education_level"]
        age = float(request.form["age"])

        education_encoded = encoder.transform([education_level])[0]

        features = pd.DataFrame(
            [[years_experience, education_encoded, age]],
            columns=["YearsExperience", "EducationLevelEncoded", "Age"]
        )
        predicted_salary = model.predict(features)[0]
        predicted_salary = round(float(predicted_salary), 2)

        return render_template(
            "index.html",
            education_options=EDUCATION_OPTIONS,
            prediction=predicted_salary,
            form_values=request.form,
        )
    except Exception as e:
        return render_template(
            "index.html",
            education_options=EDUCATION_OPTIONS,
            prediction=None,
            error=str(e),
        )


# JSON API endpoint (useful for integrating with other apps/services)
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        years_experience = float(data["years_experience"])
        education_level = data["education_level"]
        age = float(data["age"])

        education_encoded = encoder.transform([education_level])[0]
        features = pd.DataFrame(
            [[years_experience, education_encoded, age]],
            columns=["YearsExperience", "EducationLevelEncoded", "Age"]
        )
        predicted_salary = model.predict(features)[0]

        return jsonify({
            "predicted_salary": round(float(predicted_salary), 2),
            "status": "success"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
