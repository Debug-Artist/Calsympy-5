"""Optional standalone training/diagnostics script. The web app also trains at startup."""
import app

if app.MODEL is None:
    raise SystemExit(f"Training failed: {app.MODEL_ERROR}")
m = app.MODEL
print("Training complete")
print("Dataset:", m["dataset_label"])
print("Selected optimizer:", m["optimizer_name"])
print("Test metrics:")
for key in ("mse", "rmse", "mae", "r2", "training_samples", "test_samples"):
    print(f"  {key}: {m['metrics'][key]}")
print("Feature effects (display units; kg/ha per feature unit when target is tons/ha):")
for name, effect in sorted(zip(m["feature_names"], m["effects"]), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name}: {effect:.6f}")
print("Saved model:", app.MODEL_PATH)
