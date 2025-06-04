# Fall Detection Using SisFall Dataset

This project implements a binary classifier to detect human falls using wearable IMU sensor data from the SisFall dataset. It includes data preprocessing, windowing for training, and model training with CNNs.

---

## 📁 Project Structure
    dp-fall-detection/
    ├── data/
    │ ├── raw/                          # Original SisFall (.txt)
    │ ├── preprocessed/                 # Raw sensor data converted physical units (.csv)
    │ └── windows/                      # Final training input: Numpy arrays (X_windows.npy, y_labels.npy)
    ├── model/
    │ ├── checkpoints/                  # Best-performing non-DP model weights (for baseline evaluation)
    │ ├── dp_fall_detection/            # Saved DP model weights (one per ε level)
    │ ├── identity_checkpoints/         # Trained identity classifier weights
    │ ├── dp_hyperparam_tuning.py       # Optuna-based hyperparameter tuning for DP-SGD CNNs (privacy-aware fall detection)
    │ ├── evaluate_dp_model.py          # Evaluates baseline and DP-trained CNNs, logs performance metrics to CSV
    │ ├── evaluate_model.py             # Loads best model from Optuna tuning and evaluates on validation set
    │ ├── train_dp_model.py             # Main DP-SGD training pipeline (binary fall detection)
    │ ├── train_identity_model.py       # Identity inference attack model with Optuna hyperparam tuning (multi-class classification)
    │ └── train_model.py                # Non-DP CNN model training with Optuna hyperparam tuning (binary fall detection)
    ├── results/
    │ ├── dp/                           # Metrics from DP training and evaluation (.csv)
    │ └── identity/                     # Metrics from identity classifier training and evaluation (.csv)
    ├── scripts/
    │ ├── preprocess_data.py            # Converts raw sensor data (.txt) to physical units (.csv)
    │ └── prepare_training_input.py     # Slices data into windowed samples for model training input
    ├── storage/                        # Stores Optuna study for persistent hyperparameter optimization (.db)
    ├── .gitignore                      # Ignore checkpoints, data, results, and other generated files
    ├── LICENSE                         # Project license
    ├── requirements.txt                # Python dependencies (including Optuna, Opacus, Torch, etc.)
    └── README.md                       # Project documentation and usage instructions

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
