# Fall Detection Using SisFall Dataset

This project implements a binary classifier to detect human falls using wearable IMU sensor data from the SisFall dataset. It includes data preprocessing, windowing for training, and model training with CNNs.

---

## 📁 Project Structure
    dp-fall-detection/
    ├── data/
    │ ├── raw/                          # Original SisFall .txt files
    │ ├── preprocessed/                 # CSV files with converted physical units
    │ └── windows/                      # Numpy arrays (X_windows.npy, y_labels.npy)
    ├── model/
    │ └── checkpoints/                  # Saved best model weights for evaluation
    │ └── train_model.py                # CNN model training script
    ├── scripts/
    │ ├── preprocess_data.py            # Converts raw sensor data to physical units
    │ └── prepare_training_input.py     # Slices data into windows for model input
    ├── requirements.txt
    └── README.md

---

## ⚙️ Setup

```bash
# Create virtual environment
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset
Name: SisFall

Sensors: ADXL345, ITG3200, MMA8451Q

Sampling Rate: 200 Hz

Labels: Fall (1) vs. Non-Fall (0)

---

## 🛠 Features Extracted
ADXL345, ITG3200, MMA8451Q axes (x, y, z)

Window size: 200 samples (1 second)

Overlap: 50% (100 samples)

---

## 📌 Notes
Data is normalized and windowed to support differential privacy and deep learning.
