import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
MODEL_PATH = "parkinsons_cnn_brain.h5"  # path to your trained model
IMG_PATH = "patient_1.jpg"               # path to the new scan image
IMG_SIZE = (128, 128)                    # same as training

# ----------------- LOAD MODEL -----------------
print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# ----------------- PREPROCESS IMAGE -----------------
print("Loading patient image...")
img = image.load_img(IMG_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # rescale same as training
img_array = np.expand_dims(img_array, axis=0)  # shape (1,128,128,3)

# ----------------- PREDICT -----------------
prediction = model.predict(img_array)[0][0]

if prediction >= 0.5:
    label = "🧠 Parkinson Detected"
else:
    label = "✅ Healthy Brain"

print(f"\nPrediction score: {prediction:.4f}")
print(f"Result: {label}")

# ----------------- SHOW IMAGE -----------------
plt.imshow(image.load_img(IMG_PATH))
plt.title(label)
plt.axis('off')
plt.show()
