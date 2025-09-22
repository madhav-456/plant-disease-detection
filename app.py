import os
import pickle
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# ===== Lazy-load models =====
crop_model = None
fertilizer_model = None
disease_model = None


@app.route("/")
def home():
    return render_template("index.html")  # Make sure /templates/index.html exists


@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    global crop_model
    if crop_model is None:
        with open("crop_model.pkl", "rb") as f:
            crop_model = pickle.load(f)
        print("✅ Crop model loaded")
    data = request.json
    prediction = crop_model.predict([data["features"]])
    return jsonify({"prediction": prediction.tolist()})


@app.route("/predict_fertilizer", methods=["POST"])
def predict_fertilizer():
    global fertilizer_model
    if fertilizer_model is None:
        with open("fertilizer_model.pkl", "rb") as f:
            fertilizer_model = pickle.load(f)
        print("✅ Fertilizer model loaded")
    data = request.json
    prediction = fertilizer_model.predict([data["features"]])
    return jsonify({"prediction": prediction.tolist()})


@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    global disease_model
    if disease_model is None:
        disease_model = tf.keras.models.load_model("data/disease_model.h5")
        print("✅ Disease model loaded")
    data = request.json
    img_array = tf.convert_to_tensor(data["image"])
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    prediction = disease_model.predict(img_array)
    return jsonify({"prediction": prediction.tolist()})


# ===== Run server =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT
    app.run(host="0.0.0.0", port=port)  # Debug disabled for production
