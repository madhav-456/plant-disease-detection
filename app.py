from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pickle
import joblib
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EASYFARM_DIR = os.path.join(BASE_DIR, "external", "EasyFarm")

app = Flask(__name__, static_folder="static")
CORS(app, resources={r"/*": {"origins": "*"}})

# =========================
# Crop Recommendation Model
# =========================
crop_model, label_encoder = None, None

# Try loading local crop model first
try:
    crop_model = pickle.load(open(os.path.join(BASE_DIR, "crop_model.pkl"), "rb"))
    try:
        label_encoder = pickle.load(open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb"))
    except Exception:
        label_encoder = None
    print("✅ Loaded crop_model.pkl")
except Exception as e:
    print("ℹ️ Local crop_model.pkl not available:", e)

# Fallback to EasyFarm crop model if available
if crop_model is None:
    try:
        ef_crop_model_path = os.path.join(EASYFARM_DIR, "model.pkl")
        if os.path.exists(ef_crop_model_path):
            crop_model = pickle.load(open(ef_crop_model_path, "rb"))
            print("✅ Loaded EasyFarm model.pkl for crop prediction")
    except Exception as e:
        print("⚠️ Could not load EasyFarm crop model:", e)


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
        prediction = "DummyCrop"
        if crop_model is not None:
            try:
                pred = crop_model.predict(features)[0]
                # If label_encoder exists, assume model outputs index
                if label_encoder is not None:
                    prediction = label_encoder.inverse_transform([pred])[0]
                else:
                    # Many sklearn models return class label directly
                    prediction = str(pred)
            except Exception as e:
                prediction = f"Unknown ({e})"

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
    """Supports two modes:
    - JSON payload (EasyFarm ML model): expects Temperature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Phosphorous, Potassium
    - Form payload (simple rule-based fallback): expects N, P, K, crop
    """
    # Attempt EasyFarm ML path first if encoders are available
    data = request.get_json(silent=True) or {}

    # Lazy-load EasyFarm fertilizer artifacts
    global fert_model, soil_encoder, crop_encoder, fertilizer_encoder
    try:
        fert_model
    except NameError:
        fert_model = soil_encoder = crop_encoder = fertilizer_encoder = None
        try:
            fert_model_path = os.path.join(EASYFARM_DIR, "fertilizer_model.pkl")
            soil_enc_path = os.path.join(EASYFARM_DIR, "soil_encoder.pkl")
            crop_enc_path = os.path.join(EASYFARM_DIR, "crop_encoder.pkl")
            fert_enc_path = os.path.join(EASYFARM_DIR, "fertilizer_encoder.pkl")
            if all(os.path.exists(p) for p in [fert_model_path, soil_enc_path, crop_enc_path, fert_enc_path]):
                fert_model = joblib.load(fert_model_path)
                soil_encoder = joblib.load(soil_enc_path)
                crop_encoder = joblib.load(crop_enc_path)
                fertilizer_encoder = joblib.load(fert_enc_path)
                print("✅ Loaded EasyFarm fertilizer artifacts")
        except Exception as e:
            print("ℹ️ EasyFarm fertilizer artifacts not available:", e)

    try:
        # EasyFarm JSON path
        required_fields = [
            "Temperature", "Humidity", "Moisture", "Soil Type", "Crop Type",
            "Nitrogen", "Phosphorous", "Potassium"
        ]
        if data and all(k in data for k in required_fields) and fert_model is not None:
            soil_encoded = soil_encoder.transform([data['Soil Type']])[0]
            crop_encoded = crop_encoder.transform([data['Crop Type']])[0]
            features = np.array([[
                float(data['Temperature']), float(data['Humidity']), float(data['Moisture']),
                soil_encoded, crop_encoded,
                float(data['Nitrogen']), float(data['Phosphorous']), float(data['Potassium'])
            ]])
            pred_idx = fert_model.predict(features)[0]
            fertilizer_name = fertilizer_encoder.inverse_transform([pred_idx])[0]
            return jsonify({"recommended_fertilizer": str(fertilizer_name)})

        # Fallback simple rule-based using form fields
        N = float(request.form.get("N"))
        P = float(request.form.get("P"))
        K = float(request.form.get("K"))
        crop = (request.form.get("crop") or "").lower()
        if crop in fertilizer_db:
            rec = fertilizer_db[crop]
            return jsonify({
                "crop": crop,
                "ideal_N": rec["N"],
                "ideal_P": rec["P"],
                "ideal_K": rec["K"],
                "recommended_fertilizer": rec["fertilizer"]
            })
        return jsonify({"error": "Invalid input or crop not found"}), 400
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
if TENSORFLOW_AVAILABLE and os.path.exists(os.path.join(BASE_DIR, "data", "disease_model.h5")):
    try:
        disease_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "data", "disease_model.h5"))
        with open(os.path.join(BASE_DIR, "data", "disease_classes.pkl"), "rb") as f:
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
            "remedy": "⚠️ TensorFlow not available",
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
# EasyFarm-compatible Disease Prediction endpoint
# =========================
@app.route("/predict", methods=["POST"])
def predict_disease_easyfarm():
    if not disease_model:
        return jsonify({
            "prediction": "Unknown",
            "confidence": 0.0,
            "all_scores": {}
        }), 200

    if "file" not in request.files:
        return jsonify({"error": "No image uploaded (expected 'file')"}), 400

    try:
        file = request.files["file"]
        image = Image.open(file.stream)
        processed = preprocess_image(image)
        preds = disease_model.predict(processed)[0]
        idx = int(np.argmax(preds))
        disease = idx_to_class.get(idx, "Unknown")
        confidence = float(np.max(preds)) * 100.0
        all_scores = {idx_to_class.get(i, str(i)): float(p)*100.0 for i, p in enumerate(preds)}
        return jsonify({
            "prediction": disease,
            "confidence": round(confidence, 2),
            "all_scores": all_scores
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# Serve Frontend
# =========================
@app.route("/")
def index():
    # Serve EasyFarm landing page
    return send_from_directory("static", "landingpage.html")

@app.route("/detect")
def detect_page():
    return send_from_directory("static", "DiseasePrediction.html")

@app.route("/crop-recommendation")
def crop_rec_page():
    return send_from_directory("static", "Crop_recommendation.html")

@app.route("/fertilizer-recommendation")
def fert_rec_page():
    return send_from_directory("static", "Fertilizer_recommendation2.html")

@app.route("/ai-assistant")
def ai_assistant_page():
    return send_from_directory("static", "ai_assistant.html")

@app.route("/subsidy-finder")
def subsidy_page():
    return send_from_directory("static", "subsidy.html")

# PWA assets at root scope
@app.route("/service-worker.js")
def service_worker():
    return send_from_directory("static", "service-worker.js")

@app.route("/manifest.json")
def manifest():
    return send_from_directory("static", "manifest.json")


# =========================
# AI Assistant Stubs (No external keys required)
# =========================
@app.route('/api/reset', methods=['POST'])
def api_reset():
    return jsonify({
        'status': 'success',
        'message': 'Conversation reset successfully'
    })


@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get('message') or '').strip()
    if not user_message:
        return jsonify({'response': 'Please enter a question about farming.'})
    # Simple rule-based response as a fallback
    response = (
        "\n\n".join([
            f"🌱 Crop suggestion: Consider tomato or maize based on soil and weather.",
            f"🧪 Fertilizer tip: Use balanced NPK as per soil test.",
            f"🦠 Disease care: Remove infected leaves and improve airflow."
        ])
    )
    return jsonify({'response': response})


@app.route('/api/voice', methods=['POST'])
def api_voice():
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'text': '', 'response': 'No input received', 'speech_enabled': False})
    return jsonify({'text': text, 'response': 'Voice processed. ' + text, 'speech_enabled': False})


@app.route('/api/stop-speaking', methods=['POST'])
def api_stop_speaking():
    return jsonify({'status': 'success', 'message': 'Speech stopped successfully'})


@app.route('/api/tts', methods=['POST'])
def api_tts():
    return jsonify({'status': 'success', 'message': 'TTS processing (stub)'})


@app.route('/api/stop-tts', methods=['POST'])
def api_stop_tts():
    return jsonify({'status': 'success', 'message': 'TTS stopped'})


# =========================
# Run App
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


