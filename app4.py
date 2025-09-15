from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Predefined Subsidy Schemes (Example Data) ---
subsidy_schemes = [
    {"id": 1, "name": "PM-KISAN", "type": "Central Sector",
     "description": "Provides Rs.6000 annually to farmers in three installments."},
    {"id": 2, "name": "PM-KMY", "type": "Central Sector",
     "description": "Pension scheme for small and marginal farmers with monthly pension benefits."},
    {"id": 3, "name": "PMFBY", "type": "Central Sector",
     "description": "Pradhan Mantri Fasal Bima Yojana offers crop insurance against natural calamities."},
    {"id": 4, "name": "Rashtriya Krishi Vikas Yojana (RKVY)", "type": "State Scheme",
     "description": "Supports holistic agricultural development projects in states."},
    {"id": 5, "name": "Soil Health Card Scheme", "type": "Central Sector",
     "description": "Provides soil health cards to farmers to improve productivity."},
    {"id": 6, "name": "National Food Security Mission (NFSM)", "type": "Central Sector",
     "description": "Promotes sustainable increases in production of rice, wheat, pulses, and coarse cereals."},
    {"id": 7, "name": "Pradhan Mantri Krishi Sinchai Yojana (PMKSY)", "type": "Central Sector",
     "description": "Focuses on irrigation and 'More Crop Per Drop'."},
    {"id": 8, "name": "Dairy Entrepreneurship Development Scheme (DEDS)", "type": "State + Central",
     "description": "Provides subsidy for setting up small dairy units."},
    {"id": 9, "name": "Agri Infrastructure Fund (AIF)", "type": "Central Sector",
     "description": "Credit support for building post-harvest infrastructure and community farming assets."},
    {"id": 10, "name": "Neem Coated Urea Subsidy", "type": "Central Sector",
     "description": "Subsidized urea coated with neem oil to improve nutrient efficiency and reduce black marketing."}
]

@app.route("/subsidies", methods=["GET"])
def get_subsidies():
    """Return list of all subsidies"""
    print("📡 /subsidies endpoint hit")
    return jsonify({"subsidy_schemes": subsidy_schemes})

if __name__ == "__main__":
    print("✅ Subsidy Finder backend is running...")
    print(f"🔎 Available subsidies: {len(subsidy_schemes)}")
    app.run(host="0.0.0.0", port=8004, debug=True)
