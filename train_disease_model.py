import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import os

# Dataset path
DATASET_DIR = "data/plantvillage"  # put extracted dataset here

# Image preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(128, 128),
    batch_size=32,
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(128, 128),
    batch_size=32,
    subset="validation"
)

# Build CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(train_gen.class_indices), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save as Keras model
model.save("disease_model.h5")

# Save also as pickle (optional, but less common for CNNs)
with open("disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save class indices for decoding predictions
with open("disease_classes.pkl", "wb") as f:
    pickle.dump(train_gen.class_indices, f)

print("✅ Model saved: disease_model.h5 / disease_model.pkl")
