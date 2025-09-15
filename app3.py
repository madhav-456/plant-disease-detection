from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Predefined fertilizer recommendations ---
fertilizer_db = {
    "rice":      {"N": 90,  "P": 40, "K": 40, "fertilizer": "Urea (100 kg/acre), DAP (50 kg/acre), MOP (40 kg/acre)"},
    "wheat":     {"N": 120, "P": 45, "K": 45, "fertilizer": "Urea (120 kg/acre), SSP (50 kg/acre), MOP (45 kg/acre)"},
    "maize":     {"N": 85,  "P": 55, "K": 60, "fertilizer": "Urea (90 kg/acre), DAP (55 kg/acre), Potash (60 kg/acre)"},
    "sugarcane": {"N": 100, "P": 60, "K": 50, "fertilizer": "Urea (150 kg/acre), SSP (60 kg/acre), Potash (50 kg/acre)"},
    "cotton":    {"N": 100, "P": 50, "K": 50, "fertilizer": "Urea (110 kg/acre), DAP (50 kg/acre), Potash (50 kg/acre)"},
    "banana":    {"N": 60,  "P": 40, "K": 50, "fertilizer": "Urea (200 g/plant), SSP (100 g/plant), MOP (150 g/plant)"},
    "tomato":    {"N": 50,  "P": 40, "K": 50, "fertilizer": "Urea (80 kg/acre), DAP (40 kg/acre), Potash (50 kg/acre)"},
    "potato":    {"N": 110, "P": 50, "K": 45, "fertilizer": "Urea (100 kg/acre), SSP (50 kg/acre), Potash (45 kg/acre)"},
    "groundnut": {"N": 80,  "P": 40, "K": 40, "fertilizer": "Gypsum (200 kg/acre), SSP (40 kg/acre), Potash (40 kg/acre)"},
    "soybean":   {"N": 90,  "P": 60, "K": 40, "fertilizer": "DAP (60 kg/acre), SSP (40 kg/acre), Potash (40 kg/acre)"}
}

@app.route("/predict-fertilizer", methods=["POST"])
def predict_fertilizer():
    try:
        # Get form data
        N = float(request.form.get("N"))
        P = float(request.form.get("P"))
        K = float(request.form.get("K"))
        crop = request.form.get("crop", "").strip().lower()

        print(f"📥 Received Input -> N={N}, P={P}, K={K}, Crop={crop}")

        if crop in fertilizer_db:
            recommendation = fertilizer_db[crop]
            result = {
                "crop": crop.capitalize(),
                "ideal_N": recommendation["N"],
                "ideal_P": recommendation["P"],
                "ideal_K": recommendation["K"],
                "recommended_fertilizer": recommendation["fertilizer"]
            }
        else:
            result = {"error": f"❌ Crop '{crop}' not found in database"}

        print("🌱 Fertilizer Recommendation ->", result)

        # Always return JSON (simpler for frontend)
        return jsonify(result)

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("✅ Fertilizer Recommendation backend is running...")
    print("🔎 Predefined crops in DB:", list(fertilizer_db.keys()))
    app.run(host="0.0.0.0", port=2025, debug=True)
