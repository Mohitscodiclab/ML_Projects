# ==============================================
# ❤️ HEART ATTACK PREDICTION MODEL
# Author: Mohit (mohitscodiclab)
# Description: Deep Learning Model to predict
#              the likelihood of a heart attack
# ==============================================

# ---- IMPORT REQUIRED LIBRARIES ----
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# ---- LOAD YOUR DATASET ----
# Replace 'heart.csv' with your dataset path if different
df = pd.read_csv("heart.csv")

# ---- SPLIT FEATURES AND TARGET ----
# Assuming the dataset contains a column named 'target'
y = df['target']                     # Dependent variable
X = df.drop('target', axis=1)        # Independent variables

# ---- SPLIT INTO TRAINING AND TEST SETS ----
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- BUILD THE NEURAL NETWORK ----
model = Sequential()

# Input layer + first hidden layer
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.2))

# Second hidden layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

# Third hidden layer
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

# Fourth hidden layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

# Fifth hidden layer
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

# Output layer (binary classification)
model.add(Dense(1, activation='sigmoid'))

# ---- COMPILE THE MODEL ----
model.compile(
    loss='binary_crossentropy',  # Binary classification
    optimizer='adam',            # Adam optimizer for efficient learning
    metrics=['accuracy']         # Track accuracy during training
)

# ---- SAVE THE BEST MODEL BASED ON VALIDATION ACCURACY ----
checkpoint = ModelCheckpoint(
    'best_model.h5',             # File to save the best model
    monitor='val_accuracy',      # Metric to monitor
    mode='max',                  # Save model with maximum val_accuracy
    save_best_only=True,         # Save only the best version
    verbose=1
)

# ---- TRAIN THE MODEL ----
model.fit(
    X_train, y_train,
    epochs=1000,             #2000 min rakhna hi hai     # You can reduce to ~100–200 for faster testing
    batch_size=64,
    validation_split=0.1,        # 10% of training data for validation
    callbacks=[checkpoint],
    verbose=1
)

# ---- LOAD THE BEST MODEL ----
best_model = tf.keras.models.load_model('best_model.h5')

# ---- EVALUATE ON TEST DATA ----
loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Model Test Accuracy: {accuracy*100:.2f}%")

# ---- CLASSIFICATION REPORT ----
y_pred = (best_model.predict(X_test) > 0.5).astype("int32")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
