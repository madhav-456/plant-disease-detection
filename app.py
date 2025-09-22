from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pickle
import numpy as np

# =========================
# Flask App Init
# =========================
app = Flask(__name__, static_folder="static")
CORS(app, resources={r"/*": {"origins": "*"}})

# =========================
# Crop Recommendation Model
# =========================
try:
    crop_model = pickle.load(open("crop_model.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    print("✅ Crop model & encoder loaded")
except Exception as e:
    crop_model, label_encoder = None, None
    print("⚠️ Crop model not found:", e)


@app.route("/predict-crop", methods=["POST"])
def predict_crop():
    try:
        data = request.get_json()
        N = float(data.get("N"))
        P = float(data.get("P"))
        K = float(data.get("K"))
        temperature = float(data.get("temperature"))
        humidity = float(data.get("humidity"))
        ph = float(data.get("ph"))
        rainfall = float(data.get("rainfall"))

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        if crop_model and label_encoder:
            pred_idx = crop_model.predict(features)[0]
            prediction = label_encoder.inverse_transform([pred_idx])[0]
        else:
            prediction = "DummyCrop"

        return jsonify({"recommended_crop": str(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# Fertilizer Recommendation
# =========================
fertilizer_db = {
    "rice": {"N": 90, "P": 40, "K": 40, "fertilizer": "Urea 100kg/acre, DAP 50kg/acre"},
    "wheat": {"N": 120, "P": 45, "K": 45, "fertilizer": "Urea 120kg/acre, SSP 50kg/acre"},
    "maize": {"N": 85, "P": 55, "K": 60, "fertilizer": "Urea 90kg/acre, DAP 55kg/acre"},
}


@app.route("/predict-fertilizer", methods=["POST"])
def predict_fertilizer():
    try:
        N = float(request.form.get("N"))
        P = float(request.form.get("P"))
        K = float(request.form.get("K"))
        crop = request.form.get("crop").lower()

        if crop in fertilizer_db:
            rec = fertilizer_db[crop]
            return jsonify({
                "crop": crop,
                "ideal_N": rec["N"],
                "ideal_P": rec["P"],
                "ideal_K": rec["K"],
                "recommended_fertilizer": rec["fertilizer"]
            })
        else:
            return jsonify({"error": "Crop not found"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# Subsidy Finder
# =========================
subsidy_schemes = [
    {"id": 1, "name": "PM-KISAN", "type": "Central", "description": "₹6000 annually to farmers"},
    {"id": 2, "name": "PMFBY", "type": "Central", "description": "Crop insurance scheme"},
    {"id": 3, "name": "Soil Health Card", "type": "Central", "description": "Soil test and guidance"},
]


@app.route("/subsidies", methods=["GET"])
def get_subsidies():
    return jsonify({"subsidy_schemes": subsidy_schemes})


# =========================
# Disease Detection (DISABLED for Render)
# =========================
@app.route("/detect-disease", methods=["POST"])
def detect_disease():
    return jsonify({
        "disease": "Not available",
        "remedy": "⚠️ Disease detection disabled in Render deployment",
        "status": "N/A"
    })


# =========================
# Serve Frontend Pages (Clean Routes)
# =========================
@app.route("/")
def home():
    return send_from_directory("static", "index.html")

@app.route("/crop")
def crop_page():
    return send_from_directory("static", "Crop_recommendation.html")

@app.route("/fertilizer")
def fertilizer_page():
    return send_from_directory("static", "Fertilizer_recommendation.html")

@app.route("/disease")
def disease_page():
    return send_from_directory("static", "DiseasePrediction.html")

@app.route("/subsidy")
def subsidy_page():
    return send_from_directory("static", "subsidy.html")


# =========================
# Run App
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
