"""
train.py — Train a CNN on MNIST and save the model.
Run this once: python train.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF info logs

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── 1. Load MNIST ──────────────────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixels to [0, 1] and add channel dimension (grayscale = 1 channel)
x_train = x_train.astype("float32") / 255.0   # shape: (60000, 28, 28)
x_test  = x_test.astype("float32")  / 255.0   # shape: (10000, 28, 28)

x_train = np.expand_dims(x_train, -1)          # → (60000, 28, 28, 1)
x_test  = np.expand_dims(x_test,  -1)          # → (10000, 28, 28, 1)

# One-hot encode labels  e.g. 3 → [0,0,0,1,0,0,0,0,0,0]
y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test,  10)

print(f"Training samples : {x_train.shape[0]}")
print(f"Test samples     : {x_test.shape[0]}")

# ── 2. Build CNN ───────────────────────────────────────────────────────────────
model = keras.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Block 3
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),

    # Classifier head
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(10, activation="softmax"),  # 10 digit classes
], name="digit_cnn")

model.summary()

# ── 3. Compile ─────────────────────────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ── 4. Train ───────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, verbose=1),
]

history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=128,
    callbacks=callbacks,
    verbose=1,
)

# ── 5. Evaluate ────────────────────────────────────────────────────────────────
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy : {acc * 100:.2f}%")

# ── 6. Save ────────────────────────────────────────────────────────────────────
model.save("model.h5")
print("Model saved → model.h5")
