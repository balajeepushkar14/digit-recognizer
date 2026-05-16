# 🔢 Handwritten Digit Recognizer

A **Convolutional Neural Network** trained on the MNIST dataset that recognizes hand-drawn digits (0–9) in real time through a web interface.

![accuracy](https://img.shields.io/badge/Test%20Accuracy-99%25-brightgreen)
![python](https://img.shields.io/badge/Python-3.11-blue)
![tensorflow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![flask](https://img.shields.io/badge/Flask-3.0-lightgrey)

---

## 🖼️ How It Works

```
User draws digit on canvas
        ↓
Canvas → Base64 PNG → Flask /predict endpoint
        ↓
PIL resizes image to 28×28 pixels
        ↓
CNN predicts digit + confidence scores
        ↓
Result + probability bars shown to user
```

## 🏗️ CNN Architecture

| Layer | Details |
|---|---|
| Conv2D (32 filters) | 3×3 kernel, ReLU |
| BatchNorm + MaxPool | 2×2 |
| Conv2D (64 filters) | 3×3 kernel, ReLU |
| BatchNorm + MaxPool | 2×2 |
| Conv2D (128 filters) | 3×3 kernel, ReLU |
| Flatten + Dense(128) | ReLU |
| Dropout (0.4) | regularization |
| Dense(10) | Softmax output |

---

## 🚀 Run Locally

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/digit-recognizer.git
cd digit-recognizer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the CNN (takes ~5 minutes)
python train.py
# This creates model.h5

# 4. Launch the web app
python app.py

# 5. Open browser
# http://localhost:5000
```

---

## ☁️ Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` and deploys

> **Note:** First deploy trains the model (~5 min build time)

---

## 📁 Project Structure

```
digit-recognizer/
├── train.py          ← CNN training script
├── app.py            ← Flask web server
├── templates/
│   └── index.html    ← Drawing UI
├── requirements.txt
├── render.yaml       ← Render deployment config
├── Procfile
└── README.md
```

---

## 📊 Dataset

- **MNIST** — 70,000 grayscale images of handwritten digits
- 60,000 training / 10,000 testing
- Each image: 28×28 pixels, 1 channel

---

## 🛠️ Tech Stack

- **TensorFlow / Keras** — CNN model
- **Flask** — Web server & REST API
- **Pillow (PIL)** — Image preprocessing
- **HTML5 Canvas** — Drawing interface
- **Gunicorn** — Production WSGI server
