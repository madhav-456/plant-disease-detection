from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pickle
import numpy as np

# =========================
# Safe TensorFlow Import
# =========================
try:
    import tensorflow as tf
    from PIL import Image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None
    Image = None
    TENSORFLOW_AVAILABLE = False

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
# Disease Detection (Optional)
# =========================
if TENSORFLOW_AVAILABLE and os.path.exists("data/disease_model.h5"):
    try:
        disease_model = tf.keras.models.load_model("data/disease_model.h5")
        with open("data/disease_classes.pkl", "rb") as f:
            class_indices = pickle.load(f)
        idx_to_class = {v: k for k, v in class_indices.items()}
        print("✅ Disease model loaded")
    except Exception as e:
        print("⚠️ Could not load disease model:", e)
        disease_model, idx_to_class = None, {}
else:
    disease_model, idx_to_class = None, {}

remedies = {
    "Tomato_Early_blight": "Remove affected leaves, spray fungicide",
    "Potato_Late_blight": "Use copper fungicides, avoid waterlogging",
    "healthy": "Plant is healthy, continue good practices"
}


def preprocess_image(image, target_size=(128, 128)):
    img = image.convert("RGB").resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


@app.route("/detect-disease", methods=["POST"])
def detect_disease():
    if not disease_model:
        return jsonify({
            "disease": "Unknown",
            "remedy": "⚠️ TensorFlow not available on free plan",
            "status": "N/A"
        }), 200

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file = request.files["image"]
        image = Image.open(file.stream)
        processed = preprocess_image(image)
        preds = disease_model.predict(processed)[0]
        idx = int(np.argmax(preds))
        disease = idx_to_class.get(idx, "Unknown")
        remedy = remedies.get(disease, "No remedy available")
        status = "Good" if "healthy" in disease.lower() else "Bad"

        return jsonify({"disease": disease, "remedy": remedy, "status": status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# Serve Frontend
# =========================
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
