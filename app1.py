from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pickle

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ============================
# Paths
# ============================
MODEL_PATH = os.path.join("data", "disease_model.h5")
LABELS_PATH = os.path.join("data", "disease_classes.pkl")

# ============================
# Load model
# ============================
if os.path.exists(MODEL_PATH):
    print("✅ Loading trained disease model...")
    disease_model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("⚠️ No disease model found. Run train_model.py first.")
    disease_model = None

# ============================
# Load labels
# ============================
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "rb") as f:
        class_indices = pickle.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    print(f"✅ Loaded {len(idx_to_class)} disease class labels")
else:
    idx_to_class = {}
    print("⚠️ No labels found. Run train_model.py first.")

# ============================
# Remedies
# ============================
remedies = {
    "Pepper__bell___Bacterial_spot": "Use disease-free seeds, avoid overhead watering, and apply copper-based fungicides.",
    "Pepper__bell___healthy": "Your pepper plant is healthy. Maintain good watering and fertilization practices.",

    "Potato___Early_blight": "Rotate crops, remove infected leaves, and apply fungicides containing chlorothalonil or mancozeb.",
    "Potato___healthy": "Your potato plant is healthy. Maintain proper irrigation and fertilization.",
    "Potato___Late_blight": "Use copper fungicides, remove infected plants, and avoid waterlogged conditions.",

    "Tomato_Bacterial_spot": "Avoid overhead watering, use disease-free seeds, and apply copper sprays.",
    "Tomato_Early_blight": "Remove affected leaves, stake plants for air circulation, and use fungicides like mancozeb.",
    "Tomato_healthy": "Your tomato plant is healthy. Continue good farming practices.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spray with miticides or neem oil, maintain proper field hygiene.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whiteflies, remove infected plants, and use resistant varieties.",
    "Tomato_Leaf_Mold": "Improve air circulation, avoid overcrowding, and use fungicides like chlorothalonil."
}

# ============================
# Helper: preprocess image
# ============================
def preprocess_image(image, target_size=(128, 128)):
    img = image.convert("RGB").resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ============================
# Routes
# ============================
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})

@app.route("/detect-disease", methods=["POST"])
def detect_disease():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image = Image.open(file.stream)
        processed = preprocess_image(image)

        if disease_model and idx_to_class:
            preds = disease_model.predict(processed)[0]
            idx = int(np.argmax(preds))
            confidence = float(preds[idx])
            disease = idx_to_class.get(idx, "Unknown")
        else:
            disease = "Unknown"
            confidence = 0.0

        remedy = remedies.get(disease, "No remedy information available.")
        status = "Good" if "healthy" in disease.lower() else "Bad"

        print(f"📸 File: {file.filename} | 🦠 Disease: {disease} | ✅ Confidence: {confidence:.2f}")

        return jsonify({
            "disease": disease,
            "confidence": confidence,
            "remedy": remedy,
            "status": status
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/detect-disease-batch", methods=["POST"])
def detect_disease_batch():
    if "images" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("images")
    results = []

    for file in files:
        if file.filename == "":
            continue
        try:
            image = Image.open(file.stream)
            processed = preprocess_image(image)

            if disease_model and idx_to_class:
                preds = disease_model.predict(processed)[0]
                idx = int(np.argmax(preds))
                confidence = float(preds[idx])
                disease = idx_to_class.get(idx, "Unknown")
            else:
                disease = "Unknown"
                confidence = 0.0

            remedy = remedies.get(disease, "No remedy information available.")
            status = "Good" if "healthy" in disease.lower() else "Bad"

            results.append({
                "filename": file.filename,
                "disease": disease,
                "confidence": confidence,
                "remedy": remedy,
                "status": status
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return jsonify({"results": results})

# ============================
# Run server
# ============================
if __name__ == "__main__":
    app.run(debug=True, port=8000)


