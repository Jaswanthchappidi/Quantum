from flask import Flask, render_template, request, jsonify
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import joblib

from qnn_model import HybridMultiTaskQCNN
from preprocess import MODELS_DIR, FEATURE_COLUMNS

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------
# 1. Feature configuration (shared with training)
# ----------------------------------------------------
# FEATURE_COLUMNS is imported from preprocess.py
num_features = len(FEATURE_COLUMNS)

# ----------------------------------------------------
# 2. Load scaler (must be fitted on these features)
# ----------------------------------------------------
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")
scaler = joblib.load(scaler_path)

# ----------------------------------------------------
# 3. Load label mapping (id -> drug_name)
# ----------------------------------------------------
label_map_path = os.path.join(MODELS_DIR, "label_mapping.json")
if not os.path.exists(label_map_path):
    raise FileNotFoundError(f"Label mapping file not found at: {label_map_path}")

with open(label_map_path, "r", encoding="utf-8") as f:
    id_to_name = json.load(f)

num_classes = len(id_to_name)

# ----------------------------------------------------
# 4. Load model checkpoint trained with these features
# ----------------------------------------------------
model_path = os.path.join(MODELS_DIR, "best_qcnn_model_14feat.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model checkpoint not found at: {model_path}. "
                            f"Run train.py first to create it.")

checkpoint = torch.load(model_path, map_location=device)

model = HybridMultiTaskQCNN(n_features=num_features, n_classes=num_classes).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


@app.route("/")
def index():
    # index.html is using hardcoded fields, but we could pass FEATURE_COLUMNS if needed
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # -----------------------------------------------
        # Read form inputs in the exact FEATURE_COLUMNS order
        # -----------------------------------------------
        input_values = []
        for feature in FEATURE_COLUMNS:
            v = request.form.get(feature, None)
            if v is None or str(v).strip() == "":
                return jsonify({"error": f"Missing value for feature: {feature}"})
            try:
                val = float(v)
            except ValueError:
                return jsonify({"error": f"Invalid numeric value for {feature}"})
            input_values.append(val)

        # Convert to numpy and scale
        x = np.array(input_values, dtype="float32").reshape(1, -1)
        x_scaled = scaler.transform(x)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)

        # -----------------------------------------------
        # Model forward pass: returns (class_logits, conc_pred)
        # -----------------------------------------------
        with torch.no_grad():
            class_logits, conc_pred = model(x_tensor)
            probs = F.softmax(class_logits, dim=1)
            confidence, predicted = torch.max(probs, dim=1)

        pred_id = int(predicted.item())
        conf_percent = float(confidence.item() * 100.0)
        conc_value = float(conc_pred.item())

        drug_name = id_to_name.get(str(pred_id), f"Class {pred_id}")

        return jsonify(
            {
                "prediction": drug_name,
                "confidence": f"{conf_percent:.2f}%",
                "concentration": f"{conc_value:.4f} mg/L",
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
