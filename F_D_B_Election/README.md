# 🗳️ Face Detection Voting System

A secure and modern **Face Recognition-based Voting System** built using **OpenCV** and **K-Nearest Neighbors (KNN)**.  
This project replaces traditional voting methods with a **face-based authentication mechanism** and ensures vote integrity by hashing and securely storing vote data.

---

## 🚀 Features

✅ **Face Detection & Recognition** – Uses OpenCV + KNN for identifying registered users.  
✅ **Automated Voting Process** – Each vote is securely recorded in `Votes.csv`.  
✅ **Data Security** – Votes are hashed before being stored, and backups are maintained.  
✅ **User-friendly Interface** – Uses Python scripts (`add_faces.py`, `give_vote.py`) with background image support.  
✅ **Cross-platform & Extensible** – Can be improved with more ML/DL techniques.  

---

## 📂 Project Structure

```

data/
├── faces\_data.pkl        # Stored facial encodings
├── names.pkl             # Names linked with faces
├── votes\_backup.csv      # Backup of original votes
add\_faces.py               # Register new faces
background.png             # Background UI image
give\_vote.py               # Voting script
README.md                  # Documentation
requirements.txt           # Dependencies
Votes.csv                  # Main vote records (hashed)

````

---

## ⚙️ Installation

1. Clone this repository:
   ```bash[
   https://github.com/Mohitscodiclab/ML_Projects/tree/main/F_D_B_Election
   cd face-voting-system
``

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the scripts:

   * Add a face:

     ```bash
     python add_faces.py
     ```
   * Cast a vote:

     ```bash
     python give_vote.py
     ```

---

## 📦 Dependencies & Why They’re Used

### 🔹 Core Libraries

* **opencv-python (`cv2`)**

  * For **face detection & recognition**, image capturing, and preprocessing.
  * Enables camera access and handling live video feed.

* **scikit-learn**

  * Provides the **K-Nearest Neighbors (KNN)** classifier to recognize faces.
  * Lightweight ML model suitable for small to medium datasets.

* **numpy**

  * Efficient handling of image arrays and mathematical operations.
  * Used in data transformations for face recognition.

* **pickle**

  * Stores serialized data (`faces_data.pkl`, `names.pkl`) for quick loading.
  * Makes the system fast without retraining every time.

* **os**

  * Handles file system operations like reading, saving, and organizing data.

* **csv**

  * Records votes in `Votes.csv` file.
  * Also used for backup management.

* **hashlib**

  * Hashes votes to ensure **data integrity** and prevent tampering.

* **shutil**

  * Creates backups of votes before overwriting with hashed values.

* **time & datetime**

  * Timestamps for each vote.
  * Useful for logging and tracking voting sessions.

* **win32com.client (pywin32)**

  * Provides **text-to-speech (TTS)** capability on Windows.
  * Example: Announces *“Vote successfully recorded”* after voting.

---

## 🛠️ How It Works

1. **Face Registration (`add_faces.py`)**

   * Captures face images using a webcam.
   * Extracts embeddings and stores them in `faces_data.pkl`.
   * Associates each face with a name stored in `names.pkl`.

2. **Voting (`give_vote.py`)**

   * Recognizes face in real-time using KNN classifier.
   * If authenticated, the system records the vote.
   * Hashes the vote entry before saving to `Votes.csv`.
   * A backup (`votes_backup.csv`) is automatically created.

3. **Security**

   * All votes are hashed (irreversible).
   * Backup ensures data recovery in case of corruption.

---

## 🔒 Security Features

* **One-way hashing** of votes → prevents manipulation.
* **Automatic backups** before overwriting → ensures data recovery.
* **Face authentication** → prevents unauthorized voting.

---

## 🚀 Possible Improvements

🔹 **Deep Learning Models** – Replace KNN with **FaceNet** or **Dlib** for higher accuracy.

🔹 **Database Integration** – Store votes in a **secure SQL/NoSQL database** instead of CSV.

🔹 **Cross-platform TTS** – Replace `pywin32` with a platform-independent library (e.g., `pyttsx3`).

🔹 **Web-based Dashboard** – Add Flask/Django backend with a live results dashboard.

🔹 **Liveness Detection** – Prevent spoofing with photos/videos by adding eye-blink or 3D depth checks.

🔹 **Blockchain-based Voting** – For maximum transparency and tamper-proof records.

---

## 📜 Requirements

```
opencv-python
scikit-learn
numpy
pywin32
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 📸 Demo (Optional)

*Add screenshots or GIFs of the face detection and voting process here.*

---

## 👨‍💻 Author

Developed by **Mohit**
🚀 Always exploring new ideas in **AI, Cybersecurity, and Software Development**

---

## ⭐ Contribute

Pull requests are welcome!
If you’d like to suggest new features or report bugs, open an issue.

---

## 🏆 License

This project is licensed under the **MIT License** – free to use and modify.




