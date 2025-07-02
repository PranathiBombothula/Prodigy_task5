import joblib
import sys
from utils import extract_single_image_features
import json

# Load model
model = joblib.load("rf_model.pkl")

# Load calorie mapping
with open("calorie_mapping.json", "r") as f:
    calorie_dict = json.load(f)

# Get image path from command-line
image_path = sys.argv[1]

# Extract features
features = extract_single_image_features(image_path).reshape(1, -1)

# Predict
label = model.predict(features)[0]
calories = calorie_dict.get(label, "Unknown")

print(f"Predicted Food: {label}")
print(f"Estimated Calories: {calories} kcal")
