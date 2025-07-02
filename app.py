from flask import Flask, render_template, request
import joblib
import os
from utils import extract_single_image_features
import json

app = Flask(__name__)
model = joblib.load("rf_model.pkl")

with open("calorie_mapping.json", "r") as f:
    calorie_dict = json.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    calories = None
    image_path = None
    if request.method == "POST":
        image = request.files["image"]
        image_path = os.path.join("static", "uploads", image.filename)
        image.save(image_path)

        features = extract_single_image_features(image_path).reshape(1, -1)
        label = model.predict(features)[0]
        prediction = label
        calories = calorie_dict.get(label, "Unknown")

    return render_template("index.html", prediction=prediction, calories=calories, image_path=image_path)

if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    app.run(debug=True)
