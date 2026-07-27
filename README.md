# Salary Prediction — ML Project (Flask, deploy-ready)

A simple, end-to-end machine learning project that predicts an employee's
salary based on **years of experience**, **education level**, and **age**.
Built with scikit-learn's Linear Regression and served through a Flask web
app, styled with a simple form UI.

Kept intentionally low-complexity (one algorithm, three features, no
database) so it's easy to understand, extend, and deploy.

## Project structure

```
salary_prediction/
├── app.py                     # Flask web app (UI + JSON API)
├── train.py                   # Trains the model and saves it to /model
├── requirements.txt           # Python dependencies
├── Procfile                   # For Heroku / Render deployment
├── data/
│   ├── generate_data.py       # Creates the synthetic dataset
│   └── salary_data.csv        # Generated dataset (500 rows)
├── model/
│   ├── salary_model.pkl       # Trained Linear Regression model
│   └── education_encoder.pkl  # LabelEncoder for EducationLevel
└── templates/
    └── index.html             # Web form UI
```

## How it works

1. **Data**: `data/generate_data.py` creates a synthetic but realistic
   dataset (`salary_data.csv`) with 500 records. Salary is a function of
   experience, education level, and some random noise — mimicking how
   real compensation data behaves.
2. **Training**: `train.py` loads the CSV, encodes the categorical
   `EducationLevel` column, trains a `LinearRegression` model, prints
   evaluation metrics (MAE, RMSE, R²), and saves the model + encoder
   with `joblib`.
3. **Serving**: `app.py` loads the saved model/encoder once at startup
   and exposes:
   - `GET /` — a web form to enter details and see a predicted salary
   - `POST /predict` — handles the form submission
   - `POST /api/predict` — a JSON API endpoint for programmatic use

## Run it locally

```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Regenerate the dataset
python data/generate_data.py

# 4. Train the model (already trained, but you can retrain any time)
python train.py

# 5. Run the app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Using the JSON API

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"years_experience": 5, "education_level": "Masters", "age": 29}'
```

Response:
```json
{"predicted_salary": 78345.21, "status": "success"}
```

## Deploying

This project is ready to deploy on any platform that supports Python +
Flask (Render, Railway, Heroku, PythonAnywhere, Azure App Service, etc.).

### Render / Railway / Heroku (using the included Procfile)
1. Push this project to a GitHub repository.
2. Create a new **Web Service** on Render (or app on Railway/Heroku) and
   connect the repo.
3. Set the build command: `pip install -r requirements.txt`
4. Set the start command: `gunicorn app:app` (already defined in `Procfile`)
5. Deploy — the platform will install dependencies and start the app.

### Docker (optional)
If you prefer containers, here's a minimal Dockerfile you can add:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
```

## Model performance

On the synthetic dataset (80/20 train-test split):
- **MAE**: ~$4,000
- **RMSE**: ~$5,300
- **R²**: ~0.965

## Extending this project

- Swap in your own real salary dataset (keep the same column names, or
  update `train.py`/`app.py` accordingly).
- Try other models (RandomForestRegressor, XGBoost) — the pipeline
  (`train.py`) is structured so swapping the model is a one-line change.
- Add more features (job title, location, company size).
- Add input validation/rate-limiting if deploying publicly at scale.
