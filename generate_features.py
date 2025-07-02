import os
import numpy as np
from utils import extract_features_from_image
from tqdm import tqdm

image_folder = "data"  # Ensure this matches your dataset path
output_folder = "features"
os.makedirs(output_folder, exist_ok=True)

for class_name in os.listdir(image_folder):
    class_path = os.path.join(image_folder, class_name)
    if not os.path.isdir(class_path):
        continue
    for img_name in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
        img_path = os.path.join(class_path, img_name)
        try:
            features = extract_features_from_image(img_path)
            save_path = os.path.join(output_folder, f"{class_name}_{img_name}.npz")
            np.savez(save_path, features=features, label=class_name)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
