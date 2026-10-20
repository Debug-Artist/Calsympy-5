# Crop Yield Predictor

A Python/FastAPI website for a crop-yield regression project. It includes a homepage, an interactive prediction form, and an integration/results page explaining partial derivatives, MSE, vanilla gradient descent, the momentum formulation from your supplied transcript, test metrics, convergence, and feature effects.

## Files

- `app.py` — web app and model training/evaluation.
- `train_model.py` — optional training summary.
- `requirements.txt` — Python dependencies.
- `render.yaml` — Render Blueprint configuration.
- `data/crop_data.csv` — put your downloaded Kaggle CSV here.

## Run locally

Use Python 3.10 or 3.11.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`.

## Dataset configured

The uploaded `crop_yield_dataset.csv` is included at `data/crop_data.csv`. The app is configured for these columns:

```python
FEATURE_COLUMNS = [
    "Rainfall_mm",
    "Temperature_C",
    "Soil_pH",
    "Humidity_pct",
    "Fertilizer_Used_kg",
]
TARGET_COLUMN = "Yield_ton_per_ha"
TARGET_IS_TONS_PER_HECTARE = True
```

The model uses the five numeric features above and predicts `Yield_ton_per_ha`. The prediction form converts the model's tons-per-hectare output to **kg/ha** for display. Feature-effect values are also displayed in kg/ha per feature unit. Evaluation metrics are calculated in the original target units (tons/ha), so RMSE and MAE are in tons/ha and MSE is in (tons/ha)². Other CSV columns such as `Crop`, `Region`, `Soil_Type`, and `Irrigation` are currently not model inputs; the app does not use those categorical variables in this version.

The dataset has 10,000 rows. The app drops rows missing any selected feature or the target, splits the usable data into training and test sets, scales features and target for stable optimization, then compares vanilla gradient descent with the momentum formula specified in your reference. It reports test metrics and convergence history. Restart the app after changing the CSV or column configuration.

### Demo mode

If `data/crop_data.csv` is absent, the app uses a generated synthetic dataset so the pages can be previewed and deployed before the Kaggle file is ready. The pages clearly label this as demonstration data. Do not report demo metrics or feature effects as findings from real agricultural data.

## Deploy to Render

### Option A: Blueprint (recommended)

1. Push this project folder to a GitHub repository.
2. In Render, choose **New + → Blueprint** and connect the repository.
3. Render reads `render.yaml`; apply the configuration.
4. Once deployed, visit the Render-provided URL. Pages are `/`, `/crop-yield`, and `/integration`.

### Option B: Web Service

- Runtime: Python
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

No Dockerfile is required for native Python deployment. The app can run without `model_weights.npz` because it trains on startup from the CSV or the demo data.

## Important modeling notes

- The target and all features must be numeric and have clear units.
- The test split is held out from fitting; results can vary by dataset.
- The app chooses the optimizer with the lower test MSE. Momentum is not guaranteed to win.
- The momentum update follows the formula supplied in your reference transcript: `V_t = V_(t-1) + (1 - beta) * gradient`, followed by `parameter = parameter - learning_rate * V_t`. Other common momentum conventions exist.
- Feature effects from linear regression are associations, not causal proof. Rainfall, humidity, soil nutrients, and crop type may be correlated. A single linear model may not capture nonlinear agricultural relationships.
- If your dataset contains multiple crop types or regions, consider filtering to one crop/region or extending the model with categorical features.
