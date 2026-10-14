"""
app.py
Flask Web Application & REST API for Multiple Linear Regression & Feature Selection Explorer
"""
import os
import json
from flask import Flask, render_template, jsonify, request

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load precomputed model database
MODEL_DATA_PATH = os.path.join(os.path.dirname(__file__), "model_data.json")
with open(MODEL_DATA_PATH, "r", encoding="utf-8") as f:
    MODEL_DB = json.load(f)

@app.route("/")
def index():
    """Renders the main interactive regression explorer dashboard."""
    return render_template("index.html", data=MODEL_DB)

@app.route("/api/models", methods=["GET"])
def get_models():
    """Returns the complete 5-algorithm feature selection evaluation data (k=1..5)."""
    return jsonify(MODEL_DB["models"])

@app.route("/api/summary", methods=["GET"])
def get_summary():
    """Returns summary metrics across all algorithms."""
    return jsonify(MODEL_DB["summary"])

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Live inference endpoint calculating predicted profit and formula contribution breakdown.
    Expected JSON:
    {
      "algo": "Lasso",
      "k": 2,
      "rd_spend": 100000,
      "marketing_spend": 150000,
      "admin_spend": 120000,
      "state": "California"
    }
    """
    try:
        req = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400

    algo = req.get("algo", "Lasso")
    k = int(req.get("k", 2))
    k = max(1, min(5, k))

    # Match algorithm key (handle variations)
    algo_key = "Lasso"
    for key in MODEL_DB["models"].keys():
        if key.lower() in algo.lower() or algo.lower() in key.lower():
            algo_key = key
            break

    algo_info = MODEL_DB["models"][algo_key]
    eval_item = algo_info["k_evaluations"][k - 1]
    subset = eval_item["subset"]
    coefficients = eval_item["coefficients"]

    rd = float(req.get("rd_spend", 0.0))
    mkt = float(req.get("marketing_spend", 0.0))
    adm = float(req.get("admin_spend", 0.0))
    state = str(req.get("state", "California")).strip()

    state_fl = 1.0 if state.lower() == "florida" else 0.0
    state_ny = 1.0 if state.lower() in ["new york", "newyork"] else 0.0

    val_map = {
        "R&D Spend": rd,
        "Administration": adm,
        "Admin": adm,
        "Marketing Spend": mkt,
        "State_Florida": state_fl,
        "State_FL": state_fl,
        "State_New York": state_ny,
        "State_NY": state_ny
    }

    intercept_coef = coefficients[0]["beta"]
    predicted_profit = intercept_coef

    contributions = [
        {
            "feature": "Intercept (Base Constant)",
            "input_value": 1.0,
            "coef": round(intercept_coef, 4),
            "contribution": round(intercept_coef, 2),
            "percentage": 0.0
        }
    ]

    for item in coefficients[1:]:
        f_name = item["predictor"]
        coef = item["beta"]
        input_val = val_map.get(f_name, 0.0)
        contrib = coef * input_val
        predicted_profit += contrib
        contributions.append({
            "feature": f_name,
            "input_value": input_val,
            "coef": round(coef, 4),
            "contribution": round(contrib, 2),
            "percentage": 0.0
        })

    # Calculate absolute contributions share
    total_abs = sum(abs(c["contribution"]) for c in contributions) or 1.0
    for c in contributions:
        c["percentage"] = round((abs(c["contribution"]) / total_abs) * 100, 1)

    return jsonify({
        "algo": algo_info["name"],
        "k": k,
        "subset": subset,
        "predicted_profit": round(predicted_profit, 2),
        "formatted_profit": f"${predicted_profit:,.2f}",
        "intercept": round(intercept_coef, 2),
        "contributions": contributions,
        "equation": eval_item["equation"],
        "rmse": eval_item["rmse"],
        "r2": eval_item["r2"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
