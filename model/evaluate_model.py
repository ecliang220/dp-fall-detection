import csv
import os
import numpy as np
import optuna
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime

# Detect if running in Google Colab
IS_COLAB = "google.colab" in sys.modules

# Root project path for Google Colab
COLAB_ROOT = "/content/drive/MyDrive/Summer2025/CSC499/dp-fall-detection"

# Root project path for local machine (one level up from this script)
LOCAL_ROOT = Path(__file__).resolve().parents[1]

# Dynamically resolve project root based on environment 
PROJECT_ROOT = COLAB_ROOT if IS_COLAB else LOCAL_ROOT

# Training model input files: sliding windows and labels as NumPy arrays in compressed binary format (.npy)
X_PATH = PROJECT_ROOT / "data/windows/X_windows.npy"
Y_PATH = PROJECT_ROOT / "data/windows/y_labels.npy"

# Directory for saving model checkpoint files (best performing CNN weights)
MODEL_DIR_PATH = PROJECT_ROOT / "model" / "checkpoints"

# File path for trained best model
BEST_MODEL_FILE_PATH = MODEL_DIR_PATH / "best_model.pt"

# Directory for saving model evaluation metrics and Optuna optimization results
METRICS_DIR_PATH = PROJECT_ROOT / "results"

# File path for binary fall detection classifier performance metrics
BEST_MODEL_METRICS_FILE_PATH = METRICS_DIR_PATH / "best_model_metrics.csv"

# File path for binary fall detection classifier hyperparameters
BEST_MODEL_HYPERPARAMS_FILE_PATH = METRICS_DIR_PATH / "best_model_hyperparams.csv"

# Threshold applied to sigmoid output to determine binary class (1 if output > threshold, else 0)
SIGMOID_BINARY_CLASSIFICATION_THRESHOLD = 0.5

# Datetime color
GREEN = '\033[32m'
# RESET to default color
RESET = '\033[0m'

"""
Optuna Config
"""
OPTUNA_STORAGE_DIR_PATH = PROJECT_ROOT / "storage"
OPTUNA_STORAGE_PATH = f"sqlite:///{(PROJECT_ROOT / 'storage' / 'optuna_fall_detection.db').as_posix()}"
OPTUNA_STUDY_NAME = "CNN_fall_detection_optimization"

"""
Default Hyperparams
"""
DEFAULT_LAYER_COUNT = 5
DEFAULT_LEARNING_RATE = 0.003167
DEFAULT_BATCH_SIZE = 128
DEFAULT_DROPOUT = 0.3606
DEFAULT_NUM_CHANNELS = 256

def timestamp_now():
    """
    Returns the current date and time formatted as a timestamp ('[YYYY-MM-DD HH:MM:SS]'), 
    wrapped in green color for terminal output.

    Returns:
        str: A string containing the formatted timestamp wrapped in color escape codes.
    """
    return f'{GREEN}[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]{RESET}'

def flatten_and_normalize_data(X):
    """
    Normalizes sensor data across all windows using standard scaling.

    Args:
        X (np.ndarray): A 3D NumPy array of shape (num_windows, time_steps, num_features).

    Returns:
        np.ndarray: A normalized array of the same shape with zero mean and unit variance per feature.
    """
    X_flat = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler().fit(X_flat)

    return scaler.transform(X_flat).reshape(X.shape)

def load_data(x_path, y_path, batch_size):
    """
    Loads and prepares data for training and validation.

    - Loads windowed sensor data and labels from .npy files.
    - Normalizes the input features using StandardScaler.
    - Converts the arrays to PyTorch tensors.
    - Splits into training and validation datasets (80/20 split).

    Args:
        x_path (Path): Path to the input windowed feature array (.npy).
        y_path (Path): Path to the corresponding label array (.npy).
        batch_size (int): Batch size for DataLoaders.

    Returns:
        tuple: DataLoader objects for training and validation.
    """
    # Load data as NumPy arrays
    X = np.load(x_path)
    y = np.load(y_path)

    X_scaled = flatten_and_normalize_data(X)

    # Convert to torch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    return train_loader, val_loader

def make_cnn(layer_count, num_channels, dropout):
    """
    Base CNN model: Builds a 1D CNN model for fall detection using variable-depth architecture.

    - The network begins with an initial Conv1D + BatchNorm + ReLU + MaxPool block.
    - Additional convolutional blocks are added based on the specified layer_count.
    - Each additional block doubles the number of channels and includes Conv1D + BatchNorm + ReLU.
    - MaxPool1d is applied only in the first two additional blocks.
    - The output is pooled, flattened, and passed through fully connected layers ending in a single sigmoid-logit output.

    Args:
        layer_count (int): Total number of convolutional blocks to include (minimum 3).
        num_channels (int): Number of output channels for the first convolutional layer.
        dropout (float): Dropout probability applied before the final output layer.

    Returns:
        nn.Sequential: A PyTorch Sequential model ready for training.
    """
    layers = []
    input_channels = 9 # for 9 sensors
    current_channels = num_channels

    # Add the first conv block
    layers.append(nn.Conv1d(input_channels, num_channels, kernel_size=5, padding=2))
    layers.append(nn.BatchNorm1d(num_channels))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool1d(2))

    for i in range(1, layer_count):
        next_channels = current_channels * 2
        layers.append(nn.Conv1d(current_channels, next_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm1d(next_channels))
        layers.append(nn.ReLU())

        if i < 3:
            layers.append(nn.MaxPool1d(2))
        
        current_channels = next_channels

    layers.append(nn.AdaptiveAvgPool1d(1))

    model = nn.Sequential(
        *layers,
        nn.Flatten(),
        nn.Linear(current_channels, 128), 
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 1)
    )
    return model

def evaluate_model(model, val_loader):
    """
    Evaluates a trained model on a validation or test dataset.

    - Applies sigmoid activation to model outputs and classifies predictions based on SIGMOID_BINARY_CLASSIFICATION_THRESHOLD.
    - Computes standard binary classification metrics: accuracy, precision, recall, and F1 score.

    Args:
        model (nn.Module): Trained CNN model.
        val_loader (DataLoader): DataLoader for validation or test data.

    Returns:
        tuple: (accuracy, precision, recall, f1)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            X_batch = X_batch.permute(0, 2, 1)
            output = model(X_batch)
            probs = torch.sigmoid(output)
            preds = (probs > SIGMOID_BINARY_CLASSIFICATION_THRESHOLD).float()

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())

    acc = correct / total
    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return acc, precision, recall, f1

def main():
    study = optuna.load_study(
        study_name=OPTUNA_STUDY_NAME,
        storage=OPTUNA_STORAGE_PATH
    )
    
    best_params = study.best_params

    print(f"{timestamp_now()} Making CNN...")
    best_model = make_cnn(best_params['layer_count'], best_params['num_channels'], best_params['dropout'])

    print(f"{timestamp_now()} Loading model...")
    best_model.load_state_dict(torch.load(
        BEST_MODEL_FILE_PATH, 
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        weights_only=True
    ))

    print(f"{timestamp_now()} Loading dataset...")
    _, val_loader = load_data(X_PATH, Y_PATH, best_params['batch_size'])

    print(f"{timestamp_now()} Evaluating model at: {BEST_MODEL_FILE_PATH}")
    accuracy, precision, recall, f1 = evaluate_model(best_model, val_loader)

    print(f"\n{timestamp_now()} Model evaluation complete.")
    print(f"Evaluation of Best Model:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Output best model hyperparameters to CSV file
    os.makedirs(METRICS_DIR_PATH, exist_ok=True)
    with open(BEST_MODEL_HYPERPARAMS_FILE_PATH, "w", newline="") as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(["layer_count", "learning_rate", "batch_size", "num_channels", "dropout"])
        csvWriter.writerow(
            [best_params.get("layer_count", DEFAULT_LAYER_COUNT), 
             best_params.get("lr", DEFAULT_LEARNING_RATE),
             best_params.get("batch_size", DEFAULT_BATCH_SIZE), 
             best_params.get("num_channels", DEFAULT_NUM_CHANNELS), 
             best_params.get("dropout", DEFAULT_DROPOUT)]
            )

    # Output best model performance metrics to CSV file
    with open(BEST_MODEL_METRICS_FILE_PATH, "w", newline="") as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(["accuracy", "precision", "recall", "f1"])
        csvWriter.writerow([accuracy, precision, recall, f1])

if __name__ == '__main__':
    main()