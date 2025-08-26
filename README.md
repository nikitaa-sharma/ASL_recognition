# 🖐️ ASL Recognition with Deep Learning

This project is an **American Sign Language (ASL) Recognition System** built with **TensorFlow/Keras, OpenCV, and Python**. It trains a Convolutional Neural Network (CNN) on ASL datasets and uses real-time webcam input to predict hand signs (A–Z, plus `space`, `nothing`, and `del`).

---

## 📂 Project Structure

```
asl_recognition/
│── dataset/              # Training dataset (organized by class folders)
│── models/               # Saved model (.h5 file)
│── train_asl.py          # Training script
│── predict_asl.py        # Real-time prediction using webcam
│── requirements.txt      # Dependencies
│── README.md             # Documentation
```

---

## ⚙️ Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/asl_recognition.git
   cd asl_recognition
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🏋️ Training the Model

Run the training script:

```bash
python train_asl.py
```

* The model will be trained and saved as:

  ```
  models/asl_model.h5
  ```
* Training logs will display accuracy and loss per epoch.
* Final trained model can achieve **>98% validation accuracy**.

---

## 🤖 Real-Time Prediction

To run real-time ASL recognition with webcam:

```bash
python predict_asl.py
```

* A webcam window will open.
* Make a sign with your hand inside the **ROI (Region of Interest) box**.
* The model will predict the sign and display it on screen.

---

## 📊 Classes Supported

* **A–Z** (26 letters)
* **Special Tokens**: `space`, `nothing`, `del`

---

## ✅ Example Results

* `A` → ✊
* `B` → ✋
* `C` → 🤲
* `space` → Adds a space
* `del` → Deletes last character

---

## 🔮 Future Improvements

* Improve background filtering with **MediaPipe Hands** or segmentation.
* Build a **sentence builder** instead of predicting one letter at a time.
* Deploy as a **web app (Streamlit/Flask)** or **mobile app**.

---

## 👩‍💻 Author

* **Nikita Sharma**
* [GitHub](https://github.com/nikitaaa-sharma) | [LinkedIn](https://linkedin.com/in/nikita-sharma-b703852b9)

---

✨ *This project helps bridge communication barriers for the deaf and hard-of-hearing community using AI.*

---

