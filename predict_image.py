import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -------------------------
# Load Model + Labels
# -------------------------
model = load_model("Server/Model/trainedModel.h5")

with open("Server/Model/labels.pkl", "rb") as f:
    class_indices = pickle.load(f)

# Reverse mapping: index → class name
idx_to_class = {v: k for k, v in class_indices.items()}

# -------------------------
# Function to predict image
# -------------------------
def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    return idx_to_class[predicted_class], confidence

# -------------------------
# Test with an image
# -------------------------
test_img = "dataset_small/Tomato___Early_blight/your_test_image.jpg"  # change to your image path
label, conf = predict(test_img)

print(f"Prediction: {label} (Confidence: {conf:.2f})")
