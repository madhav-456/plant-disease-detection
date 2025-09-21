import os
import pickle
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# ===== Load models once at startup =====
# Crop model
try:
    with open("crop_model.pkl", "rb") as f:
        crop_model = pickle.load(f)
    print("✅ Crop model loaded")
except FileNotFoundError:
    crop_model = None
    print("⚠️ Crop model not found")

# Fertilizer model
try:
    with open("fertilizer_model.pkl", "rb") as f:
        fertilizer_model = pickle.load(f)
    print("✅ Fertilizer model loaded")
except FileNotFoundError:
    fertilizer_model = None
    print("⚠️ Fertilizer model not found")

# Disease model
try:
    disease_model = tf.keras.models.load_model("data/disease_model.h5")
    print("✅ Disease model loaded")
except Exception as e:
    disease_model = None
    print(f"⚠️ Failed to load disease model: {e}")


# ===== Routes =====
@app.route("/")
def home():
    return render_template("index.html")  # Make sure you have index.html in /templates


@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    if crop_model is None:
        return jsonify({"error": "Crop model not loaded"}), 500
    data = request.json
    # Replace with your actual prediction logic
    prediction = crop_model.predict([data["features"]])
    return jsonify({"prediction": prediction.tolist()})


@app.route("/predict_fertilizer", methods=["POST"])
def predict_fertilizer():
    if fertilizer_model is None:
        return jsonify({"error": "Fertilizer model not loaded"}), 500
    data = request.json
    # Replace with your actual prediction logic
    prediction = fertilizer_model.predict([data["features"]])
    return jsonify({"prediction": prediction.tolist()})


@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    if disease_model is None:
        return jsonify({"error": "Disease model not loaded"}), 500
    data = request.json
    # Replace with your actual image preprocessing
    img_array = tf.convert_to_tensor(data["image"])
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    prediction = disease_model.predict(img_array)
    return jsonify({"prediction": prediction.tolist()})


# ===== Run server =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT
    app.run(host="0.0.0.0", port=port, debug=True)
