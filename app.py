from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import base64
from tensorflow import keras
import os

# Disable TensorFlow debugging logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)
CORS(app) 

# Load the model
model = keras.models.load_model("model.h5")

def process_multiple_digits(base64_string):
    # Decode the Base64 string
    if "," in base64_string:
        base64_string = base64_string.split(',')[1]
        
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # 1. Find all separate shapes (contours) on the canvas
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 2. Get bounding boxes and filter out tiny dots/noise (like stray pixels)
    rects = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 50]
    
    # 3. Sort the boxes from left to right based on their X-coordinate
    rects.sort(key=lambda b: b[0])
    
    processed_digits = []
    
    for (x, y, w, h) in rects:
        cropped = img[y:y+h, x:x+w]
        
        # Resize to fit inside a 20x20 box
        if w > h:
            new_w = 20
            new_h = max(1, int((20 / w) * h))
        else:
            new_h = 20
            new_w = max(1, int((20 / h) * w))
            
        resized = cv2.resize(cropped, (new_w, new_h))
        
        # Pad to exactly 28x28
        pad_top = (28 - new_h) // 2
        pad_bottom = 28 - new_h - pad_top
        pad_left = (28 - new_w) // 2
        pad_right = 28 - new_w - pad_left
        
        final_28x28 = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, value=0
        )
        
        # Normalize and reshape
        final_input = final_28x28.astype("float32") / 255.0
        processed_digits.append(final_input.reshape(1, 28, 28, 1))
        
    return processed_digits

# --- THIS IS THE ROUTE THAT WAS MISSING ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    base64_image = data.get('image')
    
    digit_images = process_multiple_digits(base64_image)
    
    if not digit_images:
        return jsonify({'error': 'Canvas is empty'}), 400
        
    predicted_string = ""
    confidences = []
    last_probs = []
    
    # Loop through every digit found from left to right
    for img in digit_images:
        pred_array = model.predict(img)[0]
        
        # Append the predicted digit to our final string
        predicted_string += str(int(np.argmax(pred_array)))
        
        # Track confidence
        confidences.append(float(np.max(pred_array)) * 100)
        
        # We save the probability array of the final digit to keep your UI bar chart working!
        last_probs = [float(p) * 100 for p in pred_array]
        
    # Calculate the average confidence across all digits
    avg_conf = sum(confidences) / len(confidences)
    
    return jsonify({
        'digit': predicted_string,
        'confidence': avg_conf,
        'all_probs': last_probs 
    })

if __name__ == '__main__':
    print("Starting server... Make sure model.h5 is in the same folder!")
    app.run(debug=True, port=5000)
