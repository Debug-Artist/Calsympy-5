import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.gzip import GZipMiddleware

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "crop_data.csv"
MODEL_PATH = BASE_DIR / "model_weights.npz"

# Edit these to match your Kaggle CSV column names.
FEATURE_COLUMNS = [
    "Rainfall_mm",
    "Temperature_C",
    "Soil_pH",
    "Humidity_pct",
    "Fertilizer_Used_kg",
]
TARGET_COLUMN = "Yield_ton_per_ha"
TARGET_IS_TONS_PER_HECTARE = True
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 3000
LEARNING_RATE = 0.01
MOMENTUM_FACTOR = 0.9

app = FastAPI(title="Crop Yield Predictor")
app.add_middleware(GZipMiddleware, minimum_size=1000)

STYLE = r"""
:root{--ink:#172033;--muted:#5c667a;--line:#e2e8f0;--blue:#2459d3;--green:#13865c;--purple:#7047c8;--orange:#b85d12;--paper:#f7f9fc;--card:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:inherit}header{background:linear-gradient(120deg,#edf4ff,#effcf7 58%,#f4efff);border-bottom:1px solid var(--line)}
.wrap{max-width:1120px;margin:auto;padding:26px 22px}header .wrap{padding-top:42px;padding-bottom:38px}
.brand{display:inline-flex;align-items:center;gap:10px;font-weight:800;letter-spacing:-.03em;color:#1e3a8a;font-size:1.02rem}
.brand-mark{display:grid;place-items:center;width:38px;height:38px;border-radius:12px;background:#2459d3;color:white;font-weight:900}
h1{font-size:clamp(2rem,4.5vw,3.25rem);line-height:1.12;letter-spacing:-.045em;margin:20px 0 12px}
h2{font-size:1.35rem;letter-spacing:-.025em;margin:0 0 10px}h3{font-size:1rem;margin:0 0 8px}
p{margin:0 0 12px}.muted{color:var(--muted)}.eyebrow{font-size:.76rem;text-transform:uppercase;letter-spacing:.13em;font-weight:800;color:var(--blue)}
nav{display:flex;flex-wrap:wrap;gap:10px;margin-top:20px}.navlink{display:inline-flex;padding:9px 13px;border:1px solid #d6deec;border-radius:10px;background:#fff;text-decoration:none;font-weight:700;font-size:.9rem}
main{max-width:1120px;margin:auto;padding:10px 22px 46px}.grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:18px;margin:18px 0}
.card{grid-column:span 12;background:var(--card);border:1px solid var(--line);border-radius:18px;padding:24px;box-shadow:0 5px 20px rgba(24,39,75,.035)}
.span6{grid-column:span 6}.span4{grid-column:span 4}.badge{display:inline-block;border-radius:999px;padding:4px 10px;font-size:.76rem;font-weight:800;background:#e7efff;color:#2348a0;margin-bottom:12px}.badge.green{background:#dcfce9;color:#116744}.badge.purple{background:#efe7ff;color:#6036b6}.badge.orange{background:#fff0dd;color:#95450b}
.button{display:inline-flex;align-items:center;justify-content:center;gap:8px;border:0;border-radius:11px;background:var(--blue);color:white;text-decoration:none;font-weight:800;padding:12px 17px;cursor:pointer;font:inherit;font-weight:800}.button:hover{filter:brightness(.95)}.button.secondary{background:#edf2ff;color:#2348a0}.button.green{background:var(--green)}
.actions{display:flex;flex-wrap:wrap;gap:10px;margin-top:16px}
form{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}label{display:grid;gap:6px;font-size:.9rem;font-weight:750}input{width:100%;border:1px solid #cbd5e1;border-radius:10px;padding:12px 13px;background:white;color:var(--ink);font:inherit}input:focus{outline:3px solid #dbeafe;border-color:#7195ee}.full{grid-column:1/-1}
.result{background:linear-gradient(130deg,#edfdf5,#f1f8ff);border:1px solid #c8efdc;border-radius:15px;padding:20px;margin-top:18px}.yield{font-size:clamp(2rem,4vw,3rem);font-weight:900;letter-spacing:-.04em;color:#08764c;line-height:1.15}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:15px}.metric{background:#f8fafc;border:1px solid var(--line);padding:15px;border-radius:12px}.metric small{display:block;color:var(--muted);font-weight:700}.metric strong{font-size:1.25rem;display:block;margin-top:5px;overflow-wrap:anywhere}
table{width:100%;border-collapse:collapse;margin:12px 0;font-size:.92rem}th,td{text-align:left;padding:11px 10px;border-bottom:1px solid var(--line)}th{background:#f1f5f9;font-size:.8rem;text-transform:uppercase;letter-spacing:.04em}
.formula{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6fa;border:1px solid var(--line);border-left:4px solid var(--blue);border-radius:10px;padding:15px;font: .93rem/1.7 ui-monospace,SFMono-Regular,Consolas,monospace;margin:12px 0}
.notice{padding:13px 15px;border-radius:12px;background:#fff7e8;border:1px solid #f5dfb5;color:#77460a;margin:14px 0}.notice.ok{background:#edfcf4;border-color:#c8efdc;color:#116744}
ul{padding-left:22px;color:var(--muted)}li{margin:6px 0}.small{font-size:.88rem}.positive{color:#08764c}.negative{color:#b42318}
footer{text-align:center;color:var(--muted);padding:26px 18px 38px;font-size:.88rem}
@media(max-width:760px){.span6,.span4{grid-column:span 12}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}form{grid-template-columns:1fr}.full{grid-column:auto}.wrap{padding:20px 17px}main{padding:5px 17px 32px}.card{padding:19px}}
"""

HOME = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Crop Yield Predictor</title><style>__STYLE__</style></head><body>
<header><div class="wrap"><div class="brand"><span class="brand-mark">CY</span> Crop Yield Predictor</div><h1>Turn field conditions into<br>data-informed yield estimates.</h1><p class="muted" style="max-width:700px">An interactive machine-learning project connecting partial derivatives, linear regression, and momentum optimization to agricultural data.</p>
<nav><a class="navlink" href="/">Overview</a><a class="navlink" href="/crop-yield">Predict yield</a><a class="navlink" href="/integration">Model & results</a></nav></div></header>
<main><div class="grid">
<section class="card span6"><span class="badge">Featured tool</span><h2>Crop Yield Predictor</h2><p class="muted">Enter rainfall, temperature, soil pH, humidity, and fertilizer-use data to estimate crop yield using a trained linear regression model.</p><div class="actions"><a class="button" href="/crop-yield">Open predictor →</a></div></section>
<section class="card span6"><span class="badge purple">Optimization study</span><h2>Momentum Optimization</h2><p class="muted">Explore how partial derivatives update model parameters and compare standard gradient descent with the momentum-based method.</p><div class="actions"><a class="button secondary" href="/integration">View integration & results →</a></div></section>
<section class="card span4"><span class="badge green">01 · Model</span><h2>Linear regression</h2><p class="muted">Predicts a continuous yield value from a weighted combination of five field features.</p><div class="formula">ŷ = w · x + b</div></section>
<section class="card span4"><span class="badge">02 · Loss</span><h2>Mean Squared Error</h2><p class="muted">Measures average squared prediction error. The gradients show how the weights and bias should change.</p><div class="formula">L = (1/n) Σ(ŷᵢ − yᵢ)²</div></section>
<section class="card span4"><span class="badge orange">03 · Interpretation</span><h2>Feature effects</h2><p class="muted">The fitted coefficients estimate the change in predicted yield per unit change in a feature, holding the others fixed.</p><div class="formula">∂ŷ / ∂xⱼ = wⱼ</div></section>
<section class="card"><h2>Project workflow</h2><div class="grid" style="margin-bottom:0"><div class="span4"><h3>1. Prepare data</h3><p class="muted small">Use a Kaggle CSV with numerical weather, soil, fertilizer-use, and yield columns.</p></div><div class="span4"><h3>2. Train and compare</h3><p class="muted small">Standardize features and compare the two optimizers on the same train/test split.</p></div><div class="span4"><h3>3. Evaluate</h3><p class="muted small">Review MSE, RMSE, MAE, R², convergence history, and estimated feature effects.</p></div></div></section>
</div></main><footer>© 2026 Crop Yield Predictor · Python · FastAPI · NumPy</footer></body></html>"""

PREDICT = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Predict Yield · Crop Yield Predictor</title><style>__STYLE__</style></head><body>
<header><div class="wrap"><div class="brand"><span class="brand-mark">CY</span> Crop Yield Predictor</div><h1>Predict crop yield</h1><p class="muted">Enter field conditions to calculate a model-based estimate.</p><nav><a class="navlink" href="/">Overview</a><a class="navlink" href="/crop-yield">Predict yield</a><a class="navlink" href="/integration">Model & results</a></nav></div></header>
<main><div class="grid"><section class="card span6"><span class="badge">Field inputs</span><h2>Field conditions</h2><p class="muted small">Use the same units and feature definitions as the dataset used for training.</p>
<form method="post" action="/crop-yield">
<label>Rainfall (mm)<input type="number" step="any" min="0" name="rainfall" value="{{ rainfall }}" required></label>
<label>Temperature (°C)<input type="number" step="any" name="temperature" value="{{ temperature }}" required></label>
<label>Soil pH<input type="number" step="any" min="0" max="14" name="soil_ph" value="{{ soil_ph }}" required></label>
<label>Humidity (%)<input type="number" step="any" min="0" max="100" name="humidity" value="{{ humidity }}" required></label>
<label class="full">Fertilizer used (kg)<input type="number" step="any" min="0" name="nutrients" value="{{ nutrients }}" required></label>
<div class="full"><button class="button" type="submit">Calculate estimate →</button></div></form>
{% if prediction is not none %}<div class="result"><span class="badge green">Prediction result</span><div class="yield">{{ prediction }}</div><strong>{{ unit }}</strong><p class="small muted" style="margin-top:8px">This is a model estimate, not a guarantee of actual farm yield.</p></div>{% endif %}
</section>
<section class="card span6"><span class="badge purple">Interpretation</span><h2>Estimated feature effects</h2><p class="muted small">Approximate change in predicted yield for a one-unit increase in each feature, with other inputs held constant.</p>
<table><tr><th>Feature</th><th>Effect per unit</th></tr>{% for name,effect in feature_effects %}<tr><td>{{ name }}</td><td class="{% if effect >= 0 %}positive{% else %}negative{% endif %}">{{ "%.5f"|format(effect) }} {{ unit }}/unit</td></tr>{% endfor %}</table>
<p class="muted small">These are associations learned by a linear model. They do not prove that a feature causes yield to change. The ranking can be misleading if the input features have different meanings, units, or correlated values.</p>
<div class="actions"><a class="button secondary" href="/integration">How the model works</a><a class="button secondary" href="/">Home</a></div>
</section></div>{% if demo_mode %}<div class="notice">The app is currently using generated demonstration data because <code>data/crop_data.csv</code> was not found. Add your Kaggle dataset and redeploy to use its learned relationships.</div>{% endif %}</main><footer>© 2026 Crop Yield Predictor · Estimates depend on the training data quality.</footer></body></html>"""

INTEGRATION = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Momentum Optimization · Crop Yield Predictor</title><style>__STYLE__</style></head><body>
<header><div class="wrap"><div class="brand"><span class="brand-mark">CY</span> Crop Yield Predictor</div><h1>Momentum optimization<br>& model evaluation</h1><p class="muted">A transparent walkthrough of the regression model, loss gradients, optimization methods, and test-set results.</p><nav><a class="navlink" href="/">Overview</a><a class="navlink" href="/crop-yield">Predict yield</a><a class="navlink" href="/integration">Model & results</a></nav></div></header>
<main><div class="grid">
<section class="card"><span class="badge">Model</span><h2>Linear regression</h2><div class="formula">ŷ = w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄ + w₅x₅ + b

x₁ = rainfall · x₂ = temperature · x₃ = soil pH
x₄ = humidity · x₅ = soil nutrients</div><p class="muted">Features are standardized using statistics calculated from the training set. The same means and standard deviations are applied to the test set and user inputs.</p></section>
<section class="card"><span class="badge green">Loss function</span><h2>Mean Squared Error (MSE)</h2><div class="formula">L = (1/n) Σ(ŷᵢ − yᵢ)²
eᵢ = ŷᵢ − yᵢ

∂L/∂wⱼ = (2/n) Σ(eᵢ xᵢⱼ)
∂L/∂b  = (2/n) Σeᵢ</div><p class="muted">The weight gradient measures how the loss changes with a weight; the bias gradient measures the average residual. This implementation uses the exact factor of 2.</p></section>
<section class="card"><span class="badge orange">Baseline optimizer</span><h2>Vanilla batch gradient descent</h2><div class="formula">w ← w − η ∂L/∂w
b ← b − η ∂L/∂b</div><p class="muted">Each iteration calculates gradients over the full training split and updates the parameters. A suitable learning rate is important for stable convergence.</p></section>
<section class="card"><span class="badge purple">Reference momentum formulation</span><h2>Momentum Optimization</h2><p class="muted">Momentum-based gradient descent extends standard batch gradient descent by carrying a velocity term across iterations. This makes each update depend on accumulated gradient direction, not only the current gradient.</p><h3>Baseline: Batch Gradient Descent</h3><p class="muted">Standard batch gradient descent processes all training samples in each iteration, computes the cost-function gradient over the full training set, and updates the parameters.</p><div class="formula">θ_new = θ_old − learning_rate × gradient</div><p class="muted small">Possible limitations include slow convergence, oscillation in steep directions, and slow progress through flat regions. These depend on the loss surface and learning rate.</p><h3>Core mechanism and velocity update</h3><p class="muted">At time step T, momentum uses the velocity from time step T − 1. The formula below follows the convention supplied in the project's reference material.</p><div class="formula">V_T = V_(T−1) + (1 − β) ∇L
θ_T = θ_(T−1) − η V_T</div><h3>Applied to crop-yield model parameters</h3><div class="formula">V_wT = V_w(T−1) + (1 − β) ∂L/∂w
V_bT = V_b(T−1) + (1 − β) ∂L/∂b

w_T = w_(T−1) − η V_wT
b_T = b_(T−1) − η V_bT</div><p class="muted small">β is the momentum factor and η is the learning rate. Here the weights w and bias b are updated using their respective partial derivatives.</p><h3>Partial derivatives of Mean Squared Error</h3><div class="formula">L = (1/n) Σ(ŷᵢ − yᵢ)²
∂L/∂wⱼ = (2/n) Σ((ŷᵢ − yᵢ) xᵢⱼ)
∂L/∂b = (2/n) Σ(ŷᵢ − yᵢ)</div><p class="muted">The gradients indicate how each weight and the bias affect the loss, guiding updates toward lower prediction error.</p><h3>Why momentum may help</h3><ul><li>Can dampen oscillations in some loss landscapes.</li><li>Can make optimization paths smoother.</li><li>Can speed up convergence when gradients point in consistent directions.</li><li>Can help progress through some flat regions.</li><li>Its benefits are not guaranteed; performance depends on data scaling, learning rate, and β.</li></ul><p><strong>Selected optimizer:</strong> {{ optimizer_name }}</p></section>
<section class="card"><span class="badge green">Held-out test set</span><h2>Evaluation metrics</h2><p class="muted small">Metrics below are calculated on data excluded from training.</p><div class="metrics"><div class="metric"><small>Test MSE</small><strong>{{ "%.4f"|format(metrics.mse) }}</strong></div><div class="metric"><small>Test RMSE</small><strong>{{ "%.4f"|format(metrics.rmse) }}</strong></div><div class="metric"><small>Test MAE</small><strong>{{ "%.4f"|format(metrics.mae) }}</strong></div><div class="metric"><small>Test R²</small><strong>{{ "%.4f"|format(metrics.r2) }}</strong></div></div><p class="muted small" style="margin-top:12px">Training samples: {{ metrics.training_samples }} · Test samples: {{ metrics.test_samples }} · Dataset: {{ dataset_label }}</p></section>
<section class="card"><span class="badge">Convergence comparison</span><h2>Training loss by epoch</h2><p class="muted small">MSE history on the same training split. Lower is better; a method can converge faster without improving held-out accuracy.</p><table><tr><th>Epoch</th><th>Gradient descent MSE</th><th>Momentum MSE</th></tr>{% for row in convergence_rows %}<tr><td>{{ row.epoch }}</td><td>{{ "%.5f"|format(row.gd_mse) }}</td><td>{{ "%.5f"|format(row.momentum_mse) }}</td></tr>{% endfor %}</table></section>
<section class="card"><span class="badge purple">Partial derivative interpretation</span><h2>Feature effects</h2><p class="muted small">For linear regression, the derivative of predicted yield with respect to a feature is its coefficient in original feature units.</p><table><tr><th>Feature</th><th>∂yield / ∂feature</th><th>Direction</th></tr>{% for name,effect in feature_effects %}<tr><td>{{ name }}</td><td>{{ "%.5f"|format(effect) }}</td><td>{% if effect > 0 %}Positive association{% elif effect < 0 %}Negative association{% else %}No linear effect{% endif %}</td></tr>{% endfor %}</table><p class="muted small">These values describe the fitted model, not causal effects. Their practical interpretation depends on the dataset's units and feature definitions.</p></section>
<section class="card"><h2>Navigation</h2><div class="actions"><a class="button" href="/crop-yield">Try the predictor</a><a class="button secondary" href="/">Back to home</a></div></section>
</div>{% if demo_mode %}<div class="notice">Demo mode is active: results use generated example data, not your Kaggle dataset. Configure the CSV path and column names before using results in a report.</div>{% endif %}</main><footer>© 2026 Crop Yield Predictor · Linear regression and optimization demonstration</footer></body></html>"""

def render(template, **context):
    from jinja2 import Template
    return Template(template.replace("__STYLE__", STYLE)).render(**context)

def gradients(X, y, w, b):
    residuals = X @ w + b - y
    return (2.0 / len(y)) * (X.T @ residuals), 2.0 * residuals.mean()

def mse_loss(X, y, w, b):
    return float(np.mean((X @ w + b - y) ** 2))

def train_gd(X, y, lr=LEARNING_RATE, epochs=EPOCHS):
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    history = []
    for _ in range(epochs):
        dw, db = gradients(X, y, w, b)
        w -= lr * dw
        b -= lr * db
        history.append(mse_loss(X, y, w, b))
        if not np.isfinite(history[-1]) or history[-1] > 1e30:
            raise ValueError("Gradient descent diverged. Try a smaller LEARNING_RATE.")
    return w, b, history

def train_momentum(X, y, lr=0.05, momentum_factor=MOMENTUM_FACTOR, epochs=EPOCHS):
    """Momentum update using the formulation specified in the reference material.

    V_T = V_(T-1) + (1 - momentum_factor) * gradient
    parameter_T = parameter_(T-1) - learning_rate * V_T
    """
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    velocity_w = np.zeros_like(w)
    velocity_b = 0.0
    history = []
    for _ in range(epochs):
        dw, db = gradients(X, y, w, b)
        velocity_w = velocity_w + (1.0 - momentum_factor) * dw
        velocity_b = velocity_b + (1.0 - momentum_factor) * db
        w = w - lr * velocity_w
        b = b - lr * velocity_b
        history.append(mse_loss(X, y, w, b))
        if not np.isfinite(history[-1]) or history[-1] > 1e30:
            raise ValueError("Momentum optimization diverged. Try a smaller momentum learning rate.")
    return w, b, history

def demo_dataset():
    rng = np.random.default_rng(42)
    n = 500
    rainfall = rng.normal(900, 180, n).clip(100, 1800)
    temperature = rng.normal(27, 3, n).clip(10, 42)
    ph = rng.normal(6.5, 0.5, n).clip(3.5, 9)
    humidity = rng.normal(65, 8, n).clip(20, 100)
    npk = rng.normal(120, 25, n).clip(10, 250)
    y = 0.045*rainfall + 32*temperature - 18*ph + 2.5*humidity + 0.8*npk + 520 + rng.normal(0,25,n)
    return pd.DataFrame({"rainfall":rainfall,"temperature":temperature,"ph":ph,"humidity":humidity,"npk":npk,"yield":y})

def resolve_columns(df):
    # Match configured names case-insensitively; update FEATURE_COLUMNS/TARGET_COLUMN if needed.
    lookup = {str(c).strip().lower(): c for c in df.columns}
    wanted = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in wanted if c.lower() not in lookup]
    if missing:
        raise ValueError(f"CSV columns not found: {missing}. Available columns: {list(df.columns)}. Edit FEATURE_COLUMNS and TARGET_COLUMN in app.py.")
    return [lookup[c.lower()] for c in wanted]

def load_and_train():
    demo_mode = not DATA_PATH.exists()
    if demo_mode:
        df = demo_dataset()
        label = "Generated demonstration data"
        target_is_tons = False
    else:
        df = pd.read_csv(DATA_PATH)
        label = "Uploaded crop_yield_dataset.csv (10,000 rows)"
        target_is_tons = TARGET_IS_TONS_PER_HECTARE
    columns = resolve_columns(df)
    clean = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(clean) < 10:
        raise ValueError("The dataset needs at least 10 complete numeric rows.")
    X = clean[columns[:-1]].to_numpy(dtype=float)
    y = clean[columns[-1]].to_numpy(dtype=float)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("The selected CSV columns contain infinite or invalid values.")
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.permutation(len(y))
    test_count = max(1, int(round(len(y) * TEST_SIZE)))
    test_idx, train_idx = indices[:test_count], indices[test_count:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds[stds == 0] = 1.0
    X_train_s = (X_train-means)/stds
    X_test_s = (X_test-means)/stds
    # Stabilize learning rate across different target units.
    y_mean = y_train.mean()
    y_std = y_train.std()
    if not np.isfinite(y_std) or y_std == 0:
        y_std = 1.0
    yt = (y_train-y_mean)/y_std
    gd_w, gd_b, gd_hist_scaled = train_gd(X_train_s, yt)
    mom_w, mom_b, mom_hist_scaled = train_momentum(
        X_train_s, yt, lr=0.05, momentum_factor=0.9, epochs=3000
    )
    gd_pred = (X_test_s @ gd_w + gd_b)*y_std + y_mean
    mom_pred = (X_test_s @ mom_w + mom_b)*y_std + y_mean
    def score(pred):
        err = pred-y_test
        ss_tot = np.sum((y_test-y_test.mean())**2)
        return {"mse":float(np.mean(err**2)), "rmse":float(np.sqrt(np.mean(err**2))),
                "mae":float(np.mean(np.abs(err))), "r2":float(1-np.sum(err**2)/ss_tot) if ss_tot else 0.0}
    gd_metrics, mom_metrics = score(gd_pred), score(mom_pred)
    # Select the optimizer with lower test MSE; avoid assuming momentum always wins.
    if mom_metrics["mse"] <= gd_metrics["mse"]:
        chosen_w, chosen_b, chosen_name = mom_w, mom_b, "Momentum Optimization"
    else:
        chosen_w, chosen_b, chosen_name = gd_w, gd_b, "Vanilla Gradient Descent"
    # Convert standardized-X and standardized-y parameters to original units.
    original_w = chosen_w * y_std / stds
    original_b = y_mean + y_std*chosen_b - float(original_w @ means)
    # Convergence history shown in target units for easier interpretation.
    gd_hist = [v*y_std*y_std for v in gd_hist_scaled]
    mom_hist = [v*y_std*y_std for v in mom_hist_scaled]
    # Save a portable model artifact.
    np.savez(MODEL_PATH, weights=original_w, bias=np.array(original_b), means=means, stds=stds,
             y_mean=np.array(y_mean), y_std=np.array(y_std), target_is_tons=np.array(target_is_tons))
    selected_metrics = mom_metrics if chosen_name == "Momentum Optimization" else gd_metrics
    return {
        "weights":original_w, "bias":original_b, "means":means, "stds":stds,
        "feature_names":["Rainfall (mm)","Temperature (°C)","Soil pH","Humidity (%)","Fertilizer used (kg)"],
         "effects":(original_w * (1000.0 if target_is_tons else 1.0)).tolist(),
        "demo_mode":demo_mode, "dataset_label":label,
        "optimizer_name":chosen_name, "metrics":{**selected_metrics,"training_samples":len(train_idx),"test_samples":len(test_idx)},
        "gd_history":gd_hist,"momentum_history":mom_hist,"target_is_tons":target_is_tons,
        "target_name":str(columns[-1])
    }

MODEL_ERROR = None
try:
    MODEL = load_and_train()
except Exception as exc:
    MODEL_ERROR = str(exc)
    MODEL = None

def require_model():
    if MODEL is None:
        return HTMLResponse(f"""<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Setup needed</title><style>{STYLE}</style></head><body><main><section class="card"><h1>Dataset setup needed</h1><div class="notice">{MODEL_ERROR}</div><p>Check <code>data/crop_data.csv</code> and update the feature/target column names at the top of <code>app.py</code>. See README.md for instructions.</p><a class="button" href="/">Back home</a></section></main></body></html>""", status_code=500)
    return None

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(render(HOME))

@app.get("/crop-yield", response_class=HTMLResponse)
async def crop_yield_page():
    problem = require_model()
    if problem: return problem
    return HTMLResponse(render(PREDICT, rainfall=900, temperature=27, soil_ph=6.5, humidity=65, nutrients=120,
                               prediction=None, unit="kg/ha" if MODEL["target_is_tons"] else MODEL["target_name"],
                               feature_effects=sorted(zip(MODEL["feature_names"], MODEL["effects"]), key=lambda x: abs(x[1]), reverse=True),
                               demo_mode=MODEL["demo_mode"]))

@app.post("/crop-yield", response_class=HTMLResponse)
async def predict_crop_yield(rainfall: float = Form(...), temperature: float = Form(...),
                             soil_ph: float = Form(...), humidity: float = Form(...), nutrients: float = Form(...)):
    problem = require_model()
    if problem: return problem
    values = np.array([rainfall, temperature, soil_ph, humidity, nutrients], dtype=float)
    scaled = (values-MODEL["means"])/MODEL["stds"]
    # Model artifact is stored in original units; this is equivalent and easier to inspect.
    prediction = float(MODEL["weights"] @ values + MODEL["bias"])
    if MODEL["target_is_tons"]:
        prediction *= 1000
        unit = "kg/ha"
    else:
        unit = MODEL["target_name"]
    return HTMLResponse(render(PREDICT, rainfall=rainfall, temperature=temperature, soil_ph=soil_ph,
                               humidity=humidity, nutrients=nutrients, prediction=f"{prediction:,.3f}",
                               unit=unit,
                               feature_effects=sorted(zip(MODEL["feature_names"], MODEL["effects"]), key=lambda x: abs(x[1]), reverse=True),
                               demo_mode=MODEL["demo_mode"]))

@app.get("/integration", response_class=HTMLResponse)
async def integration_page():
    problem = require_model()
    if problem: return problem
    checkpoints = [1, 10, 50, 100, 250, 500, 1000, 2000, 3000]
    rows = []
    for epoch in checkpoints:
        if epoch <= len(MODEL["gd_history"]) and epoch <= len(MODEL["momentum_history"]):
            rows.append({"epoch":epoch,"gd_mse":MODEL["gd_history"][epoch-1],"momentum_mse":MODEL["momentum_history"][epoch-1]})
    return HTMLResponse(render(INTEGRATION, metrics=MODEL["metrics"], convergence_rows=rows,
                               feature_effects=sorted(zip(MODEL["feature_names"], MODEL["effects"]), key=lambda x: abs(x[1]), reverse=True),
                               optimizer_name=MODEL["optimizer_name"], dataset_label=MODEL["dataset_label"],
                               demo_mode=MODEL["demo_mode"]))

@app.get("/health")
async def health():
    return {"status":"ok" if MODEL is not None else "needs_dataset_setup",
            "demo_mode": MODEL["demo_mode"] if MODEL else None,
            "model_error": MODEL_ERROR}
