from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import os
import pickle
import numpy as np

# Optional: if you want deep learning model support
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# -------- CONFIG --------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXT = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# -------- LOAD MODELS (if available) --------
crop_model, label_encoder, disease_model, class_labels = None, None, None, None

try:
    with open("model.pkl", "rb") as f:
        crop_model = pickle.load(f)
    with open("crop_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    print("✅ Crop model + encoder loaded.")
except Exception as e:
    print("⚠️ Crop model/encoder not found. Using fallback. Error:", e)

try:
    disease_model = load_model("disease_model.h5")
    with open("disease_classes.pkl", "rb") as f:
        disease_classes = pickle.load(f)
    class_labels = {v: k for k, v in disease_classes.items()}
    print("✅ Disease model loaded.")
except Exception as e:
    print("⚠️ Disease model not found. Using demo fallback. Error:", e)


# -------- AI Assistant (Chatbot) --------
faq_data = {
    "fertilizer": "For leafy crops, use nitrogen-rich fertilizers like urea. For fruiting crops, add potassium-based fertilizers.",
    "pest": "Use neem oil spray or organic pesticides to control common pests.",
    "irrigation": "Drip irrigation saves water and improves crop yield compared to flood irrigation.",
    "wheat": "Wheat grows best in cool weather with well-drained loamy soil.",
    "rice": "Rice requires standing water and clayey soil with proper drainage."
}

@app.route("/ask-ai", methods=["POST"])
def ask_ai():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Message is required"}), 400

    user_message = data["message"].lower()

    # Rule-based farming answers
    for keyword, answer in faq_data.items():
        if keyword in user_message:
            return jsonify({"reply": answer})

    # Fallback response if no match
    return jsonify({"reply": f"I’m not sure about that. Please consult your local agriculture officer for: {user_message}"})


# -------- Disease Detection (image upload) --------
@app.route("/detect-disease", methods=["POST"])
def detect_disease():
    if "image" not in request.files:
        return jsonify({"error": "Upload image with field name 'image'."}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Upload JPG/PNG under 5MB."}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    try:
        img = Image.open(path).convert("RGB").resize((128, 128))
        arr = np.array(img) / 255.0
        arr = arr.reshape(1, 128, 128, 3)

        if disease_model and class_labels:
            pred = disease_model.predict(arr)
            idx = int(np.argmax(pred))
            disease_name = class_labels[idx]
            confidence = float(np.max(pred))
            return jsonify({"disease": disease_name, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # fallback demo
    return jsonify({
        "disease": "Leaf Blight",
        "confidence": 0.8,
        "remedies": ["Remove infected leaves", "Spray fungicide", "Maintain proper irrigation"]
    })


# -------- Crop Recommendation --------
@app.route("/predict-crop", methods=["POST"])
def predict_crop():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Provide soil/climate details"}), 400

    try:
        features = np.array([[data.get("N", 0), data.get("P", 0), data.get("K", 0),
                              data.get("temperature", 25), data.get("humidity", 50),
                              data.get("ph", 6.5), data.get("rainfall", 100)]])
        if crop_model and label_encoder:
            pred = crop_model.predict(features)
            crop = label_encoder.inverse_transform(pred)[0]
            return jsonify({"recommended_crop": crop})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # fallback demo
    return jsonify({"recommended_crop": "Rice"})


# -------- Run Server --------
if __name__ == "__main__":
    app.run(debug=True)
