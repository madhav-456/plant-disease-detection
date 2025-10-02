import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import pickle
import os

# Path to small dataset
dataset_path = "dataset_small"  # adjust if you named it differently

# -------------------------
# Data Generators
# -------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),  # MobileNetV2 input size
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# -------------------------
# Load Pre-trained MobileNetV2
# -------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

# Freeze base layers
base_model.trainable = False

# -------------------------
# Add Custom Classification Layers
# -------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# -------------------------
# Compile Model
# -------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------
# Train Model
# -------------------------
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,   # fewer epochs for demo
    verbose=1
)

# -------------------------
# Save Model + Labels
# -------------------------
os.makedirs("Server/Model", exist_ok=True)

# Save architecture
model_json = model.to_json()
with open("Server/Model/trainedModel.json", "w") as json_file:
    json_file.write(model_json)

# Save *full* model (better than just weights)
model.save("Server/Model/trainedModel.h5")

# Save labels (class mapping)
with open("Server/Model/labels.pkl", "wb") as f:
    pickle.dump(train_generator.class_indices, f)

print("✅ Model training complete. Files saved in Server/Model/:")
print(" - trainedModel.json (architecture)")
print(" - trainedModel.h5   (full model)")
print(" - labels.pkl        (class mapping)")
