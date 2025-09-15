from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Try loading your model
try:
    model = pickle.load(open(r"C:\Users\MADHAVAN\OneDrive\miniprojectfarm\model.pkl", "rb"))
    print("✅ Model loaded successfully:", type(model))
except Exception as e:
    model = None
    print("⚠️ Model not found. Using dummy recommendations. Error:", e)

@app.route("/predict-crop", methods=["POST"])
def predict_crop():
    try:
        data = request.get_json()

        # Convert all inputs to float (allowing decimal values)
        N = float(data.get("N"))
        P = float(data.get("P"))
        K = float(data.get("K"))
        temperature = float(data.get("temperature"))
        humidity = float(data.get("humidity"))
        ph = float(data.get("ph"))
        rainfall = float(data.get("rainfall"))

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Print received inputs
        print("📥 Received Input ->",
              f"N={N}, P={P}, K={K}, Temp={temperature}, Humidity={humidity}, pH={ph}, Rainfall={rainfall}")

        if model:
            prediction = model.predict(features)[0]
        else:
            prediction = "DummyCrop"

        # Print predicted output
        print("🌱 Predicted Crop ->", prediction)

        return jsonify({"recommended_crop": str(prediction)})

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500

# --- Pre-estimated test values for different crops ---
def test_predefined_values():
    test_samples = {
        "Rice":       [90, 40, 40, 25.5, 85.2, 6.5, 200],
        "Cotton":     [100, 50, 50, 35.0, 60.0, 7.0, 80],
        "Wheat":      [120, 45, 45, 20.0, 65.0, 6.8, 120],
        "Banana":     [60, 40, 50, 28.0, 90.0, 6.2, 200],
        "Maize":      [85, 55, 60, 27.0, 70.0, 6.5, 150],
        "Sugarcane":  [100, 60, 50, 30.0, 80.0, 6.8, 250],
        "Tomato":     [50, 40, 50, 22.0, 65.0, 6.0, 100],
        "Potato":     [110, 50, 45, 18.0, 70.0, 5.5, 120],
        "Groundnut":  [80, 40, 40, 26.0, 60.0, 6.2, 90]
    }

    if model:
        print("\n🔎 Running predefined test predictions:\n")
        for crop, values in test_samples.items():
            features = np.array([values])
            prediction = model.predict(features)[0]
            print(f"➡️ Expected {crop} | Predicted: {prediction}")
    else:
        print("⚠️ No model found, skipping predefined tests.")

if __name__ == "__main__":
    test_predefined_values()
    app.run(host="0.0.0.0", port=2025, debug=True)
