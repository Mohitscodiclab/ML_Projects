## ❤️ **Heart Attack Prediction System**

### *ML-based Health Risk Assessment Tool*

Author: **Mohit ([@mohitscodiclab](https://github.com/mohitscodiclab))**

---

### 🧠 **Overview**

This project uses **Deep Learning (TensorFlow/Keras)** to predict the likelihood of a **heart attack** based on a patient's medical data.

It consists of two main parts:

1. **Model Training Script (`heart_attack_prediction.py`)**
   – Trains a neural network using the UCI Heart Disease dataset.
2. **Prediction Script (`mini_heart_doctor.py`)**
   – Loads the trained model and allows doctors/patients to enter medical details for instant risk analysis.
   

---

### 🩺 **Key Features**

✅ Predicts heart attack risk using trained neural network
✅ Saves & reloads the best model (`best_model.h5`)
✅ Interactive patient input system
✅ Explains each medical input in simple language
✅ Prints confidence percentage and doctor-style advice
✅ Extendable to GUI / Web App (Tkinter or Streamlit)

---

### 🧩 **Folder Structure**

```
Heart_Attack_Prediction/
│
├── heart_attack_prediction.py        # Training the model
├── mini_Heart_Doctor.py                    # Prediction system (user input)
├── heart.csv                         # Dataset (UCI Heart Disease)
├── best_model.h5                     # Saved best model
└── README.md                         # Project documentation
```

---

### ⚙️ **Setup Instructions**

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/mohitscodiclab/Heart_Attack_Prediction.git
cd Heart_Attack_Prediction
```

#### 2️⃣ Install Dependencies

```bash
pip install tensorflow scikit-learn pandas numpy
```

#### 3️⃣ Add Dataset

Download the dataset from the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
and save it as:

```
heart.csv
```

Make sure it contains a `target` column (0 = No disease, 1 = Disease).

---

### 🧮 **1. Train the Model**

Run:

```bash
python heart_attack_prediction.py
```

This script:

* Splits your data into training & test sets
* Builds a deep learning model
* Trains it over 2000 epochs (adjustable)
* Saves the **best performing model** as `best_model.h5`

✅ After training, you’ll see test accuracy and a classification report.

---

### 🔍 **2. Run the Predictor**

Once training is done, run:

```bash
python heart_attack_predictor.py
```

You’ll be asked to enter **patient medical details** such as:

* Age, sex
* Chest pain type
* Blood pressure
* Cholesterol
* Blood sugar
* ECG results
* Heart rate
* Exercise-induced angina
* ST depression (oldpeak)
* Slope, number of vessels, thalassemia

After entering details, you’ll get output like:

```
----------------------------------------------
🔍 Prediction Result:
⚠️ HIGH RISK of Heart Attack (82.34% confidence)

🩺 Advice:
- Immediate cardiologist consultation recommended.
- Maintain diet and exercise routine.
- Monitor cholesterol and blood pressure regularly.
----------------------------------------------
```

---

### 📊 **Model Architecture**

| Layer | Type   | Units | Activation | Dropout |
| ----- | ------ | ----- | ---------- | ------- |
| 1     | Dense  | 128   | ReLU       | 0.2     |
| 2     | Dense  | 64    | ReLU       | 0.2     |
| 3     | Dense  | 32    | ReLU       | 0.2     |
| 4     | Dense  | 64    | ReLU       | 0.2     |
| 5     | Dense  | 32    | ReLU       | 0.2     |
| 6     | Output | 1     | Sigmoid    | —       |

**Loss Function:** Binary Crossentropy
**Optimizer:** Adam
**Metric:** Accuracy
**Checkpoint:** `best_model.h5` (saved at best validation accuracy)

---

### 🧬 **Medical Feature Reference**

| Feature      | Description                    | Range / Notes                                        |
| ------------ | ------------------------------ | ---------------------------------------------------- |
| **Age**      | Age in years                   | Numeric                                              |
| **Sex**      | 1 = Male, 0 = Female           | Binary                                               |
| **cp**       | Chest pain type (0–3)          | 0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic |
| **trestbps** | Resting BP (mm Hg)             | Normal: 120                                          |
| **chol**     | Cholesterol (mg/dl)            | Normal <200, High ≥240                               |
| **fbs**      | Fasting blood sugar >120 mg/dl | 1=Yes, 0=No                                          |
| **restecg**  | Resting ECG                    | 0=Normal, 1=ST abnormality, 2=LV hypertrophy         |
| **thalach**  | Max heart rate achieved        | Typically 120–200                                    |
| **exang**    | Exercise induced angina        | 1=Yes, 0=No                                          |
| **oldpeak**  | ST depression                  | 0.0–6.0 (higher → worse)                             |
| **slope**    | Slope of ST segment            | 0=Upsloping, 1=Flat, 2=Downsloping                   |
| **ca**       | Major vessels (0–3)            | More vessels = higher risk                           |
| **thal**     | Thalassemia                    | 1=Normal, 2=Fixed defect, 3=Reversible defect        |

---

### 🧠 **Example Test Inputs**

#### ✅ Low Risk

```
Age: 35
Sex: 0
Chest pain type: 0
Resting blood pressure: 118
Cholesterol: 190
Fasting blood sugar > 120: 0
Resting ECG results: 0
Max heart rate achieved: 170
Exercise induced angina: 0
ST depression (oldpeak): 0.0
Slope: 0
Number of major vessels (ca): 0
Thalassemia (thal): 1
```

#### ⚠️ High Risk

```
Age: 63
Sex: 1
Chest pain type: 3
Resting blood pressure: 150
Cholesterol: 290
Fasting blood sugar > 120: 1
Resting ECG results: 2
Max heart rate achieved: 120
Exercise induced angina: 1
ST depression (oldpeak): 3.0
Slope: 2
Number of major vessels (ca): 2
Thalassemia (thal): 3
```

---

### 🩸 **How the Prediction Works**

The neural network takes 13 medical inputs and outputs a probability (0–1):

* **>0.5** → High Risk (⚠️)
* **≤0.5** → Low Risk (✅)

Confidence (%) = model’s prediction probability × 100

---

### 💡 **Future Enhancements**

* Add **Tkinter GUI** for user-friendly data entry
* Convert to **Streamlit web app** for online access
* Store patient records with timestamp
* Integrate live ECG/IoT data input
* Add explainable AI module (feature importance)

---

### 🔒 **Disclaimer**

This model provides **statistical predictions**, not medical diagnosis.
Always consult a **qualified healthcare professional** before taking any medical action.

---

### 🧑‍💻 **Credits**

* **Author:** [Mohit Kumar](https://github.com/mohitscodiclab)
* **Dataset:** [UCI Machine Learning Repository – Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
* **Frameworks:** TensorFlow, Keras, Scikit-learn, Pandas, NumPy

