"""
app.py — Flask web server for the Digit Recognizer.
"""

import io
import base64
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Load once at startup
print("Loading model …")
model = keras.models.load_model("model.h5")
print("Model ready ✓")

def preprocess(image_data: str) -> np.ndarray:
    """
    Convert base64 PNG (from <canvas>) → 28×28 float32 array
    ready for the CNN.
    """
    # Strip the data-URL prefix ("data:image/png;base64,…")
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    raw = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(raw)).convert("L")   # grayscale

    # Canvas: white background, black stroke → invert so digit is white on black
    img = ImageOps.invert(img)

    img = img.resize((28, 28), Image.LANCZOS)

    arr = np.array(img, dtype="float32") / 255.0      # [0, 1]
    arr = arr.reshape(1, 28, 28, 1)                    # batch of 1
    return arr


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    image_data = data.get("image", "")

    if not image_data:
        return jsonify({"error": "No image data"}), 400

    arr = preprocess(image_data)
    probs = model.predict(arr, verbose=0)[0]   # shape (10,)

    digit      = int(np.argmax(probs))
    confidence = float(np.max(probs)) * 100

    # Return all probabilities for the bar chart
    all_probs = {str(i): float(probs[i]) * 100 for i in range(10)}

    return jsonify({
        "digit":      digit,
        "confidence": round(confidence, 2),
        "all_probs":  all_probs,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
