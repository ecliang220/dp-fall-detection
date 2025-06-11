"""
model_util.py

This module provides core utilities for training, evaluating, and configuring CNN-based models 
for fall detection and identity inference using wearable IMU sensor data. It defines dataset 
paths, model architectures, evaluation functions, and hyperparameter management tools for use 
in privacy-preserving machine learning experiments (including differential privacy via CLM-DP).

Key functionalities:
- CNN model definitions for fall detection and identity inference.
- Evaluation functions for binary and multi-class classifiers.
- Data loading and normalization.
- Hyperparameter reading and formatting.
- Path definitions for datasets, models, metrics, and experiment logs.
- Constants for model configuration and Optuna tuning ranges.
"""
import sys
import random
import csv
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

from util.util import print_color_text, print_color_text_with_timestamp

# --------------------------------------------------------------------
# Environment Config
# --------------------------------------------------------------------
# Detect if running in Google Colab
IS_COLAB = "google.colab" in sys.modules
# Root project path for Google Colab
COLAB_ROOT = "/content/drive/MyDrive/Summer2025/CSC499/dp-fall-detection"
# Root project path for local machine (one level up from this script)
LOCAL_ROOT = Path(__file__).resolve().parents[1]
# Dynamically resolve project root based on environment 
PROJECT_ROOT = COLAB_ROOT if IS_COLAB else LOCAL_ROOT

# --------------------------------------------------------------------
# Data Details
# --------------------------------------------------------------------
# Number of Adult participants
NUM_ADULTS = 23
# Number of Elder participants
NUM_ELDERS = 15

# Fixed-length sliding window parameters
WINDOW_SIZE = 200           # 200 rows = 1 second at 200 Hz
WINDOW_INTERVAL = 100       # 50% overlap (stride = 100)

# Directory containing Preprocessed .csv Timeseries Data
PREPROCESSED_DIR_PATH = PROJECT_ROOT / "data/preprocessed"
# Directory Path for Training Windows
WINDOWS_DIR_PATH = PROJECT_ROOT / "data/windows"

# Labels for FALL and IDENTITY windows 
WINDOWS_FILE_NAME = "X_windows.npy"
FALL_LABELS_FILE_NAME = "y_fall_labels.npy"
IDENTITY_LABELS_FILE_NAME = "y_identity_labels.npy"
X_PATH = WINDOWS_DIR_PATH / WINDOWS_FILE_NAME
Y_FALL_PATH = WINDOWS_DIR_PATH / FALL_LABELS_FILE_NAME
Y_IDENTITY_PATH = WINDOWS_DIR_PATH / IDENTITY_LABELS_FILE_NAME

# --------------------------------------------------------------------
# Model Config
# --------------------------------------------------------------------
# Number of epochs to train the CNN model during each trial
NUM_EPOCHS = 30
# Number of epochs to wait without F1 improvement before stopping early
EARLY_STOPPING_PATIENCE = 5
# Threshold for classifying sigmoid output (0.0–1.0) as binary class
SIGMOID_BINARY_CLASSIFICATION_THRESHOLD = 0.5
# Random seed for reproducibility
RANDOM_SEED = 42
# Number of classes for training IDENTITY inference classifier
NUM_CLASSES = 38 # Model will output 38 class scores

# Default Hyperparameter Values to fallback on
DEFAULT_LAYER_COUNT = 5
DEFAULT_LEARNING_RATE = 0.003167
DEFAULT_BATCH_SIZE = 128
DEFAULT_DROPOUT = 0.3606
DEFAULT_NUM_CHANNELS = 256

# --------------------------------------------------------------------
# Model Directory and File Paths/Names
# --------------------------------------------------------------------
# Directory path for saved model files (best performing CNN weights)
MODEL_DIR_PATH = PROJECT_ROOT / "model"
# Directory path for Fall Detection binary classifer checkpoints
FALL_MODEL_DIR_PATH =  MODEL_DIR_PATH / "fall_checkpoints"
# File path for trained best Fall model
BEST_FALL_MODEL_FILE_PATH = FALL_MODEL_DIR_PATH / "best_fall_model.pt"
# Directory path for Identity Inference multi-class classifer checkpoints
IDENTITY_MODEL_DIR_PATH =  MODEL_DIR_PATH / "identity_checkpoints"
# File path for trained best Identity model
BEST_IDENTITY_MODEL_FILE_PATH = IDENTITY_MODEL_DIR_PATH / "best_identity_model.pt"

# --------------------------------------------------------------------
# Results/Metrics Directory and File Paths/Names
# --------------------------------------------------------------------
# Directory for model evaluation metrics and Optuna optimization results
METRICS_DIR_PATH = PROJECT_ROOT / "results"

# Directory for saving FALL model evaluation metrics and Optuna optimization results
FALL_METRICS_DIR_PATH = PROJECT_ROOT / "results"/ "fall"
# File path for FALL classifier performance metrics
BEST_FALL_MODEL_METRICS_FILE_PATH = FALL_METRICS_DIR_PATH / "best_fall_model_metrics.csv"
# File path for best FALL model hyperparameters from Optuna tuning
BEST_FALL_MODEL_HYPERPARAMS_FILE_PATH = FALL_METRICS_DIR_PATH / "best_fall_model_hyperparams.csv"

# Directory for saving IDENTITY model evaluation metrics and Optuna optimization results
IDENTITY_METRICS_DIR_PATH = PROJECT_ROOT / "results"/ "identity"
# File path for IDENTITY classifier performance metrics
BEST_IDENTITY_MODEL_METRICS_FILE_PATH = IDENTITY_METRICS_DIR_PATH / "best_identity_model_metrics.csv"
# File path for best IDENTITY model hyperparameters from Optuna tuning
BEST_IDENTITY_MODEL_HYPERPARAMS_FILE_PATH = IDENTITY_METRICS_DIR_PATH / "best_identity_model_hyperparams.csv"

# Directory Path for Output Logs
LOGS_DIR_PATH = PROJECT_ROOT / "logs"
# File Path for Log of Windows with Noise Injection Completed
LOG_FILE_PATH = LOGS_DIR_PATH / "noise_injection_log.csv"

# Directory for model evaluation metrics on CLM-DP data
CLM_DP_METRICS_DIR_PATH = METRICS_DIR_PATH / "clm_dp"
# File path for CLM-DP evaluation metrics
CLM_DP_EVAL_METRICS_FILE_PATH = CLM_DP_METRICS_DIR_PATH / "clm_dp_eval_metrics.csv"

# --------------------------------------------------------------------
# Fall Detection Classifier Optuna Config
# --------------------------------------------------------------------
OPTUNA_STORAGE_DIR_PATH = PROJECT_ROOT / "storage"

OPTUNA_FALL_STORAGE_PATH = f"sqlite:///{str(Path(PROJECT_ROOT) / 'storage' / 'optuna_fall_detection.db')}"
OPTUNA_FALL_RESULTS_FILE_PATH = METRICS_DIR_PATH / "optuna_fall_results.csv"
OPTUNA_FALL_STUDY_NAME = "CNN_fall_detection_optimization"
OPTUNA_FALL_N_TRIALS = 30 # more trials to run
OPTUNA_FALL_LR_MIN = 1e-4
OPTUNA_FALL_LR_MAX = 1e-2
OPTUNA_FALL_DROPOUT_MIN = 0.3
OPTUNA_FALL_DROPOUT_MAX = 0.6
OPTUNA_FALL_BATCH_SIZE_VALS = [16, 32, 64, 128]
OPTUNA_FALL_NUM_CHANNELS_VALS = [64, 128, 256]
OPTUNA_FALL_LAYERS_MIN = 3
OPTUNA_FALL_LAYERS_MAX = 5

# --------------------------------------------------------------------
# Identity Inference Classifier Optuna Config
# --------------------------------------------------------------------
OPTUNA_IDENTITY_STORAGE_PATH = f"sqlite:///{str(Path(PROJECT_ROOT) / 'storage' / 'optuna_identity_inference.db')}"
OPTUNA_IDENTITY_RESULTS_FILE_PATH = METRICS_DIR_PATH / "optuna_identity_results.csv"
OPTUNA_IDENTITY_STUDY_NAME = "CNN_identity_classification_optimization"
OPTUNA_IDENTITY_N_TRIALS = 30 # more trials to run
OPTUNA_IDENTITY_LR_MIN = 1e-4
OPTUNA_IDENTITY_LR_MAX = 1e-2
OPTUNA_IDENTITY_DROPOUT_MIN = 0.3
OPTUNA_IDENTITY_DROPOUT_MAX = 0.6
OPTUNA_IDENTITY_BATCH_SIZE_VALS = [16, 32, 64, 128]
OPTUNA_IDENTITY_NUM_CHANNELS_VALS = [64, 128, 256]
# OPTUNA_IDENTITY_NUM_CHANNELS_VALS = [64, 128, 256, 512]
OPTUNA_IDENTITY_LAYERS_MIN = 3
OPTUNA_IDENTITY_LAYERS_MAX = 6

class CLM_DP_Experiment:
    """
    Configuration class for the Correlated Laplace Mechanism (CLM) Differential Privacy experiment.

    Attributes:
        EPSILON_VALS (list of float): Privacy budgets (ε) to evaluate, from weakest to strongest DP.
        DECAY_FACTOR (float): Controls temporal correlation in the noise; higher means stronger correlation.
        JITTER_EPS (float): Small value added to diagonal of covariance matrix to ensure positive-definiteness.
    """
    # --------------------------------------------------------------------
    # Experiment: CLM Differential Privacy Config
    # --------------------------------------------------------------------
    # Epsilon values to test (weakest to strongest)
    EPSILON_VALS = [8.0, 4.0, 2.0, 1.0, 0.5]
    # Controls strength of temporal correlation; higher = stronger correlation across time steps (0 < decay < 1)
    DECAY_FACTOR = 0.95
    # Numerical jitter added to the diagonal to ensure positive-definiteness
    JITTER_EPS = 1e-6

def set_seed(seed=RANDOM_SEED):
    """
    Sets random seeds for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

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

def get_params_dict_from_csv(csv_file_path):
    """
    Extracts and returns model hyperparameters from a CSV file as a dictionary.

    The CSV file should have the following format:
        - The first row (header) contains the keys: 'layer_count', 'learning_rate', 'batch_size', 
          'num_channels', and 'dropout'.
        - The second row contains the corresponding values for these keys.

    Args:
        csv_file_path (str): Path to the CSV file containing model hyperparameters.

    Returns:
        dict: A dictionary with the model parameters ('layer_count', 'learning_rate', 'batch_size', 
              'num_channels', 'dropout') converted to appropriate data types.
    """
    params = dict()
    with open(csv_file_path, 'r') as csv_file:
        csv_dict_reader = csv.DictReader(csv_file)
        params = next(csv_dict_reader, {})
        params = {
            "layer_count": int(params["layer_count"]),
            "learning_rate": float(params["learning_rate"]),
            "batch_size": int(params["batch_size"]),
            "num_channels": int(params["num_channels"]),
            "dropout": float(params["dropout"]),
        }
    return params

def print_model_metrics(accuracy, precision, recall, f1, summary_title="Final Evaluation of Best Model"):
    print_color_text_with_timestamp(f"{summary_title}:", "BRIGHT_MAGENTA")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print_color_text(f"F1 Score: {f1:.4f}", "RED")


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
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    return train_loader, val_loader


def make_fall_detection_cnn(layer_count, num_channels, dropout):
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

def evaluate_fall_detection_model(model, val_loader):
    """
    Evaluates a trained CNN model on a validation or test dataset.

    - Applies sigmoid activation to model outputs and classifies predictions using a fixed binary threshold.
    - Compares predictions against ground truth to compute accuracy, precision, recall, and F1 score.
    - Assumes binary classification with labels in {0, 1} and outputs in [0, 1] after sigmoid.

    Args:
        model (nn.Module): Trained CNN model for binary classification.
        val_loader (DataLoader): DataLoader containing the validation or test dataset.

    Returns:
        tuple:
            A tuple of (accuracy, precision, recall, f1), each as a float.
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

def make_identity_inference_cnn(layer_count, num_channels, dropout):
    """
    Base CNN model: Builds a 1D CNN model for identity classification using variable-depth architecture.

    - The network begins with an initial Conv1D + BatchNorm + ReLU + MaxPool block.
    - Additional convolutional blocks are added based on the specified layer_count.
    - Each additional block doubles the number of channels and includes Conv1D + BatchNorm + ReLU.
    - MaxPool1d is applied only in the first two additional blocks.
    - The output is pooled, flattened, and passed through fully connected layers ending in a vector of class logits.

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
        nn.Linear(128, NUM_CLASSES) # Output logits: one per participant class (0–37)
    )
    return model

def evaluate_identity_inference_model(model, val_loader):
    """
    Evaluates a trained identity classifier on a validation or test dataset.

    - Applies argmax to model output logits to select the most likely class.
    - Computes multi-class classification metrics: accuracy, precision, recall, and F1 score.

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
            preds = torch.argmax(output, dim=1) # Predicts most likely class by selecting index with highest logit

            correct += (preds == y_batch.view(-1)).sum().item()
            total += y_batch.size(0)

            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())

    acc = correct / total
    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()

    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    return acc, precision, recall, f1
