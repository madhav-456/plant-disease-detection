import os
import shutil
import random

# Paths
RAW_DIR = r"C:\Users\MADHAVAN\OneDrive\miniprojectfarm\data\PlantVillage\PlantVillage"
OUTPUT_DIR = r"C:\Users\MADHAVAN\OneDrive\miniprojectfarm\data"

TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

# Make output dirs
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

SPLIT_RATIO = 0.8  # 80% train, 20% test

for class_name in os.listdir(RAW_DIR):
    class_path = os.path.join(RAW_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Create class dirs
    train_class_dir = os.path.join(TRAIN_DIR, class_name)
    test_class_dir = os.path.join(TEST_DIR, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Copy files
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))
    for img in test_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(test_class_dir, img))

print("✅ Dataset split into train/ and test/")
