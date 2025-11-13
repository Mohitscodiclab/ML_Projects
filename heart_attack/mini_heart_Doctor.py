# ==============================================
# ❤️ HEART ATTACK PREDICTION USING TRAINED MODEL
# Author: Mohit (mohitscodiclab)
# Description: Loads trained model and predicts
#              the likelihood of a heart attack
# ==============================================

import tensorflow as tf
import numpy as np
import pandas as pd

# ---- LOAD THE TRAINED MODEL ----
model = tf.keras.models.load_model('best_model.h5')

print("\n==============================================")
print(" 🩺 ADVANCED HEART ATTACK RISK PREDICTION SYSTEM ")
print("==============================================\n")

# ---- ASK USER FOR PATIENT DETAILS ----
# These features match the heart dataset columns
# (based on the UCI Heart Disease dataset)
def get_patient_data():
    print("Please enter the patient's medical details below:\n")

    # 1️⃣ Age
    age = float(input("👤 Age (years): "))

    # 2️⃣ Sex
    print("\n⚧️ Sex options:\n  1 = Male\n  0 = Female")
    sex = int(input("Enter sex (1/0): "))

    # 3️⃣ Chest Pain Type
    print("\n💓 Chest Pain Type:")
    print("  0 = Typical Angina        → Chest pain related to decreased blood supply")
    print("  1 = Atypical Angina       → Chest pain not related to heart")
    print("  2 = Non-anginal Pain      → Pain not related to heart (e.g., muscle pain)")
    print("  3 = Asymptomatic          → No pain but signs of heart disease")
    cp = int(input("Enter chest pain type (0–3): "))

    # 4️⃣ Resting Blood Pressure
    trestbps = float(input("\n🩸 Resting blood pressure (in mm Hg, normal 120): "))

    # 5️⃣ Cholesterol Level
    print("\n🥓 Cholesterol Level (mg/dl):")
    print("  • Normal: <200")
    print("  • Borderline High: 200–239")
    print("  • High: ≥240")
    chol = float(input("Enter serum cholesterol: "))

    # 6️⃣ Fasting Blood Sugar
    print("\n🧪 Fasting Blood Sugar > 120 mg/dl:")
    print("  1 = Yes (High sugar, may indicate diabetes)")
    print("  0 = No (Normal sugar)")
    fbs = int(input("Enter 1 or 0: "))

    # 7️⃣ Resting ECG Results
    print("\n🫀 Resting ECG Results:")
    print("  0 = Normal")
    print("  1 = ST-T wave abnormality (possible ischemia)")
    print("  2 = Left ventricular hypertrophy (thick heart muscle)")
    restecg = int(input("Enter ECG result (0–2): "))

    # 8️⃣ Maximum Heart Rate Achieved
    thalach = float(input("\n🏃‍♂️ Maximum heart rate achieved (e.g., 120–200): "))

    # 9️⃣ Exercise Induced Angina
    print("\n😣 Exercise Induced Angina:")
    print("  1 = Yes (pain during exercise → possible heart problem)")
    print("  0 = No (no pain during exercise)")
    exang = int(input("Enter 1 or 0: "))

    # 🔟 ST Depression
    print("\n📉 ST Depression (oldpeak):")
    print("  - The amount of depression in the ST segment of the ECG.")
    print("  - Higher values (1.0–6.0) indicate more severe heart stress.")
    oldpeak = float(input("Enter ST depression value (e.g., 0.0–6.0): "))

    # 11️⃣ Slope of Peak Exercise ST Segment
    print("\n📈 Slope of Peak Exercise ST Segment:")
    print("  0 = Upsloping   → Better recovery, often normal")
    print("  1 = Flat        → Mild abnormality")
    print("  2 = Downsloping → Strong indicator of heart disease")
    slope = int(input("Enter slope type (0–2): "))

    # 12️⃣ Number of Major Vessels Colored by Fluoroscopy
    print("\n🔢 Number of Major Vessels (0–3):")
    print("  - The more vessels, the higher the heart disease risk.")
    ca = int(input("Enter number of major vessels (0–3): "))

    # 13️⃣ Thalassemia (Thal)
    print("\n🧬 Thalassemia (Thal) Type:")
    print("  1 = Normal (no defect)")
    print("  2 = Fixed Defect (permanent heart issue)")
    print("  3 = Reversible Defect (temporary blood flow issue)")
    thal = int(input("Enter thalassemia type (1–3): "))

    # Combine all features into a single array
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    return features


# ---- GET INPUT FROM USER ----
patient_data = get_patient_data()

# ---- PREDICT USING TRAINED MODEL ----
prediction = model.predict(patient_data)
probability = float(prediction[0][0])

# ---- INTERPRET RESULT ----
print("\n----------------------------------------------")
print("🔍 Prediction Result:")
if probability > 0.5:
    print(f"⚠️ HIGH RISK of Heart Attack ({probability*100:.2f}% confidence)")
    print("\n🩺 Advice:")
    print("- Immediate cardiologist consultation recommended.")
    print("- Consider lifestyle changes: diet, exercise, stress reduction.")
    print("- Monitor blood pressure and cholesterol levels regularly.")
else:
    print(f"✅ LOW RISK of Heart Attack ({(1 - probability)*100:.2f}% confidence)")
    print("\n💪 Advice:")
    print("- Maintain healthy lifestyle and diet.")
    print("- Regular exercise and check-ups.")
    print("- Keep monitoring blood sugar and cholesterol.")
print("----------------------------------------------\n")

print("🧠 Note: This prediction is based on statistical modeling.\n"
      "Always consult a certified doctor for professional diagnosis.\n")
