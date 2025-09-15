from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import pickle

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ====================================================
# Crop Recommendation (from app2.py)
# ====================================================
try:
    crop_model = pickle.load(open("model.pkl", "rb"))
    print("✅ Crop model loaded:", type(crop_model))
except Exception as e:
    crop_model = None
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
        prediction = crop_model.predict(features)[0] if crop_model else "DummyCrop"

        return jsonify({"recommended_crop": str(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ====================================================
# Disease Detection (from app1.py) with Fallback
# ====================================================
try:
    import tensorflow as tf
    from PIL import Image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None
    Image = None
    TENSORFLOW_AVAILABLE = False

if TENSORFLOW_AVAILABLE and os.path.exists("data/disease_model.h5"):
    try:
        print("✅ Loading disease model...")
        disease_model = tf.keras.models.load_model("data/disease_model.h5")
    except Exception as e:
        print("⚠️ Could not load disease model:", e)
        disease_model = None
else:
    disease_model = None
    print("⚠️ TensorFlow not available — disease detection disabled.")

remedies = {
    "Pepper__bell___Bacterial_spot": "Use disease-free seeds, avoid overhead watering, and apply copper-based fungicides.",
    "Potato___Early_blight": "Rotate crops, remove infected leaves, and apply fungicides containing chlorothalonil or mancozeb.",
    "Potato___Late_blight": "Use copper fungicides, remove infected plants, and avoid waterlogged conditions.",
    "Tomato_Bacterial_spot": "Avoid overhead watering, use disease-free seeds, and apply copper sprays.",
    "Tomato_Early_blight": "Remove affected leaves, stake plants for air circulation, and use fungicides like mancozeb.",
    "Tomato_Leaf_Mold": "Improve air circulation, avoid overcrowding, and use fungicides like chlorothalonil.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spray with miticides or neem oil, maintain proper field hygiene.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whiteflies, remove infected plants, and use resistant varieties."
}


@app.route("/detect-disease", methods=["POST"])
def detect_disease():
    if not disease_model:
        return jsonify({
            "disease": "Unknown",
            "confidence": 0.0,
            "remedy": "TensorFlow not available on free plan",
            "status": "N/A"
        }), 200

    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    try:
        image = Image.open(file.stream)
        img = image.convert("RGB").resize((128, 128))
        img_array = np.array(img) / 255.0
        processed = np.expand_dims(img_array, axis=0)

        preds = disease_model.predict(processed)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])
        disease = "Unknown"
        if idx in remedies:
            disease = list(remedies.keys())[idx]
        remedy = remedies.get(disease, "No remedy info available.")
        status = "Good" if "healthy" in disease.lower() else "Bad"

        return jsonify({"disease": disease, "confidence": confidence, "remedy": remedy, "status": status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ====================================================
# Fertilizer Recommendation (from app3.py)
# ====================================================
fertilizer_db = {
    "rice": {"N": 90, "P": 40, "K": 40, "fertilizer": "Urea (100 kg/acre), DAP (50 kg/acre), MOP (40 kg/acre)"},
    "wheat": {"N": 120, "P": 45, "K": 45, "fertilizer": "Urea (120 kg/acre), SSP (50 kg/acre), MOP (45 kg/acre)"},
    "maize": {"N": 85, "P": 55, "K": 60, "fertilizer": "Urea (90 kg/acre), DAP (55 kg/acre), Potash (60 kg/acre)"},
    "sugarcane": {"N": 100, "P": 60, "K": 50, "fertilizer": "Urea (150 kg/acre), SSP (60 kg/acre), Potash (50 kg/acre)"},
    "cotton": {"N": 100, "P": 50, "K": 50, "fertilizer": "Urea (110 kg/acre), DAP (50 kg/acre), Potash (50 kg/acre)"},
    "banana": {"N": 60, "P": 40, "K": 50, "fertilizer": "Urea (200 g/plant), SSP (100 g/plant), MOP (150 g/plant)"},
    "tomato": {"N": 50, "P": 40, "K": 50, "fertilizer": "Urea (80 kg/acre), DAP (40 kg/acre), Potash (50 kg/acre)"},
    "potato": {"N": 110, "P": 50, "K": 45, "fertilizer": "Urea (100 kg/acre), SSP (50 kg/acre), Potash (45 kg/acre)"},
    "groundnut": {"N": 80, "P": 40, "K": 40, "fertilizer": "Gypsum (200 kg/acre), SSP (40 kg/acre), Potash (40 kg/acre)"},
    "soybean": {"N": 90, "P": 60, "K": 40, "fertilizer": "DAP (60 kg/acre), SSP (40 kg/acre), Potash (40 kg/acre)"}
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
                "crop": crop.capitalize(),
                "ideal_N": rec["N"], "ideal_P": rec["P"], "ideal_K": rec["K"],
                "recommended_fertilizer": rec["fertilizer"]
            })
        else:
            return jsonify({"error": "Crop not found"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ====================================================
# Subsidy Finder (from app4.py)
# ====================================================
subsidy_schemes = [
    {"id": 1, "name": "PM-KISAN", "type": "Central Sector", "description": "Provides Rs.6000 annually to farmers."},
    {"id": 2, "name": "PM-KMY", "type": "Central Sector", "description": "Pension scheme for small farmers."},
    {"id": 3, "name": "PMFBY", "type": "Central Sector", "description": "Crop insurance scheme."},
    {"id": 4, "name": "Soil Health Card Scheme", "type": "Central Sector", "description": "Provides soil health cards."},
    {"id": 5, "name": "PMKSY", "type": "Central Sector", "description": "Irrigation and 'More Crop Per Drop'."}
]


@app.route("/subsidies", methods=["GET"])
def get_subsidies():
    return jsonify({"subsidy_schemes": subsidy_schemes})


# ====================================================
# Run the Unified App
# ====================================================
if __name__ == "__main__":
    print("✅ Unified FarmAI backend running...")
    app.run(host="0.0.0.0", port=2025, debug=True)
