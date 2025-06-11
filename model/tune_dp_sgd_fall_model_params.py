"""
dp-sgd_hyperparam_tuning.py

Tunes and evaluates CNN models for fall detection using IMU sensor data with differential privacy (DP-SGD via Opacus).
Uses Optuna to optimize hyperparameters for improved performance under a fixed privacy budget (ε).

Logs all results to CSV and saves the best model and metrics for reproducible research in privacy-preserving ML.

Author: Ellie Liang
Date: 2025-06-03
"""
import csv
from datetime import datetime
import os

import optuna
# Tells PyTorch to allow more flexible GPU memory allocation (helps prevent CUDA out of memory errors)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path
from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore", message="Secure RNG turned off.*")
warnings.filterwarnings("ignore", message="Optimal order is the largest alpha.*")

"""
When True, Opacus will use a cryptographically secure random number generator (CSPRNG) for 
adding Gaussian noise in DP-SGD to ensure noise is unpredictable and suitable for privacy 
guarantees under adversarial threat models.

TODO: Turn on for final training run and reporting official ε values in paper/documentation
"""
SECURE_MODE = False

"""
Opacus DP-SGD Config
"""
# Default ε for tuning
EPSILON_VAL = 4.0 
# Delta value: Failure probability (acceptable risk of privacy leakage)
DELTA_VAL = 1e-5
# Gradient clipping: maximum L2 norm for per-sample gradients
CLIPPING_NORM = 1.0

"""
Environment Config
"""
# Detect if running in Google Colab
IS_COLAB = "google.colab" in sys.modules
# Root project path for Google Colab
COLAB_ROOT = "/content/drive/MyDrive/Summer2025/CSC499/dp-fall-detection"
# Root project path for local machine (one level up from this script)
LOCAL_ROOT = Path(__file__).resolve().parents[1]
# Dynamically resolve project root based on environment 
PROJECT_ROOT = COLAB_ROOT if IS_COLAB else LOCAL_ROOT

"""
Directory and File Paths
"""
# Training model input files: windowed IMU data and binary labels in .npy format
X_PATH = PROJECT_ROOT / "data/windows/X_windows.npy"
Y_PATH = PROJECT_ROOT / "data/windows/y_labels.npy"
# Directory for saving DP model checkpoint files (best performing CNN weights)
MODEL_DIR_PATH = PROJECT_ROOT / "model"
# Directory for saving DP model fall classifiers
DP_MODEL_DIR_PATH = MODEL_DIR_PATH / "dp_fall_detection"
# Directory for model evaluation metrics and Optuna optimization results
METRICS_DIR_PATH = PROJECT_ROOT / "results"
# Directory for DP model evaluation metrics
DP_METRICS_DIR_PATH = METRICS_DIR_PATH / "dp"
# File path for trained best dp model
BEST_MODEL_FILE_PATH = DP_MODEL_DIR_PATH / "best_dp_model.pt"
# File path for best DP binary fall detection classifier performance metrics
BEST_MODEL_METRICS_FILE_PATH = METRICS_DIR_PATH / "best_dp_model_metrics.csv"
# File path for best DP model hyperparameters from Optuna tuning
BEST_MODEL_HYPERPARAMS_FILE_PATH = METRICS_DIR_PATH / "best_dp_model_hyperparams.csv"

"""
Training Config
"""
# Random seed for reproducibility across data splits, weight init, and tuning
RANDOM_SEED = 42
# Number of training epochs per model
NUM_EPOCHS = 30
# Number of epochs to wait without F1 improvement before stopping early
EARLY_STOPPING_PATIENCE = 5
# Threshold for classifying sigmoid output (0.0–1.0) as binary class
SIGMOID_BINARY_CLASSIFICATION_THRESHOLD = 0.5
# Expanded alpha values for tighter RDP-based epsilon accounting
# ALPHAS = [1 + x / 10.0 for x in range(1, 200)] + list(range(21, 128))
# Output for invalid performance metrics
INVALID_METRIC = "invalid"
# Output for Not Applicable performance metrics
NAN_METRIC = float("nan")

"""
Terminal Output Color Adjustments
"""
# Datetime color
GREEN = '\033[32m'
# Epoch label color
BLUE = '\033[34m'
# IMPORTANT label color
RED = '\033[31m'
# Task label color
PURPLE = '\033[35m'
# Section (Study/Evaluation) label color
BRIGHT_MAGENTA = '\033[95m'
# Miscellaneous label color
BRIGHT_BLUE = '\033[94m'
# RESET to default color
RESET = '\033[0m'

"""
Default Hyperparameters (used when Optuna-derived parameters are unavailable)
"""
DEFAULT_LAYER_COUNT = 5
DEFAULT_LEARNING_RATE = 0.003167
# DEFAULT_BATCH_SIZE = 128
DEFAULT_BATCH_SIZE = 64 # Decreased to fit cuda memory
DEFAULT_DROPOUT = 0.3606
# DEFAULT_NUM_CHANNELS = 256
DEFAULT_NUM_CHANNELS = 64 # Decreased to fit cuda memory

"""
Optuna Config
"""
OPTUNA_STORAGE_DIR_PATH = PROJECT_ROOT / "storage"
OPTUNA_STORAGE_PATH = f"sqlite:///{str(Path(PROJECT_ROOT) / 'storage' / 'dp_optuna_fall_detection.db')}"
OPTUNA_RESULTS_FILE_PATH = DP_METRICS_DIR_PATH / "dp_optuna_results.csv"

OPTUNA_STUDY_NAME = "dp_CNN_fall_detection_optimization"
OPTUNA_N_TRIALS = 30 # more trials to run
OPTUNA_LR_MIN = 1e-4
OPTUNA_LR_MAX = 3e-3
OPTUNA_DROPOUT_MIN = 0.1
OPTUNA_DROPOUT_MAX = 0.5
OPTUNA_CLIPPING_NORM_MIN = 0.5
OPTUNA_CLIPPING_NORM_MAX = 2.0
OPTUNA_NUM_CHANNELS_VALS = [32, 64, 128]
OPTUNA_LAYERS_MIN = 3
OPTUNA_LAYERS_MAX = 5

# Training start time for version control
training_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# Initial number of trials completed when Optuna study is loaded
initial_completed = 0

def color_text(text, color_code):
    """
    Wraps a given string with ANSI escape codes to display colored text in the terminal.

    Args:
        text (str): The text to be colored.
        color_code (str): The ANSI escape code representing the desired text color.

    Returns:
        str: The input text wrapped with the provided color code and a reset code to return to default styling.
    """
    return f'{color_code}{text}{RESET}'

def print_timestamp_now():
    """
    Returns the current date and time formatted as a timestamp ('[YYYY-MM-DD HH:MM:SS]'), 
    wrapped in green color for terminal output.

    Returns:
        str: A string containing the formatted timestamp wrapped in color escape codes.
    """
    return color_text(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]", GREEN)

def remove_opacus_prefix(opacus_state_dict):
    """
    Converts an Opacus-wrapped state_dict to a standard PyTorch format.

    - Removes the '_module.' prefix added by Opacus when wrapping models for DP-SGD.
    - Ensures compatibility when loading saved DP models into standard nn.Module architectures.

    Args:
        opacus_state_dict (OrderedDict): A state dictionary potentially containing '_module.' prefixes in parameter keys.

    Returns:
        OrderedDict: A cleaned state dictionary with all '_module.' prefixes removed.
    """
    new_state_dict = OrderedDict()
    for k, v in opacus_state_dict.items():
        if k.startswith("_module."):
            new_key = k[len("_module."):]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

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


def compute_class_balance(train_loader):
    """
    Computes the class imbalance ratio (negative / positive) 
    from the training dataset to be used as pos_weight for BCEWithLogitsLoss.

    Args:
        train_loader (DataLoader): DataLoader containing the training set.

    Returns:
        float: Ratio of negative to positive samples.
    """
    pos = 0
    neg = 0
    for _, labels in train_loader:
        pos += (labels == 1).sum().item()
        neg += (labels == 0).sum().item()
    return neg / pos if pos > 0 else 1.0


def set_seed(seed=42):
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

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    sample_rate = batch_size / len(train_set)
    train_loader = DataLoader(
        train_set,
        batch_sampler=UniformWithReplacementSampler(
            num_samples=len(train_set),
            sample_rate=sample_rate,
        ),
        num_workers=0,  # TODO: Increase if memory allows
    )
    val_loader = DataLoader(val_set, batch_size=batch_size)

    return train_loader, val_loader

def make_cnn(layer_count, num_channels, dropout):
    """
    Constructs a 1D CNN for fall detection using privacy-compatible layers.

    - Stacks convolutional blocks with GroupNorm, ReLU, and optional MaxPool1d.
    - Doubles the number of channels after each convolutional block.
    - Applies global average pooling, dropout, and a final linear layer.
    - Output is a single logit (no activation), intended for BCEWithLogitsLoss.

    Args:
        layer_count (int): Number of convolutional blocks (minimum 3).
        num_channels (int): Number of filters in the first convolutional layer.
        dropout (float): Dropout rate before the final output layer.

    Returns:
        nn.Sequential: The constructed CNN model.
    """
    layers = []
    input_channels = 9 # for 9 sensors
    current_channels = num_channels

    # Add the first conv block
    layers.append(nn.Conv1d(input_channels, num_channels, kernel_size=5, padding=2))
    layers.append(nn.GroupNorm(1, current_channels))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool1d(2))

    for i in range(1, layer_count):
        next_channels = current_channels * 2
        layers.append(nn.Conv1d(current_channels, next_channels, kernel_size=3, padding=1))
        layers.append(nn.GroupNorm(1, next_channels))
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

def train_model(model, train_loader, val_loader, learning_rate, epochs=NUM_EPOCHS, all_metrics=True,
                use_dp=False, target_epsilon=8.0, delta=1e-5, clipping_norm=1.0):
    """
    Trains the CNN model on fall detection data with optional differential privacy using Opacus.

    - Uses BCEWithLogitsLoss with class imbalance correction via pos_weight.
    - Applies Adam optimizer and learning rate scheduling based on validation F1 score.
    - If `use_dp` is True, applies DP-SGD using Opacus with a specified target ε, δ, and clipping norm.
    - Performs early stopping if no F1 improvement is observed for a fixed patience period.
    - Tracks training and validation metrics: accuracy, precision, recall, F1, and loss.

    Args:
        model (nn.Module): The CNN model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        learning_rate (float): Learning rate for the Adam optimizer.
        epochs (int, optional): Maximum number of training epochs. Defaults to NUM_EPOCHS.
        all_metrics (bool, optional): If True, returns all performance metrics and model state.
        use_dp (bool, optional): If True, trains with differential privacy using Opacus.
        target_epsilon (float, optional): Target privacy budget ε for DP-SGD (required if use_dp is True).
        delta (float, optional): Target failure probability δ for DP-SGD (required if use_dp is True).
        clipping_norm (float, optional): Maximum gradient norm for DP-SGD clipping.

    Returns:
        tuple or float:
            If use_dp and all_metrics are True:
                Returns (val_loss, accuracy, precision, recall, f1, actual_epsilon, noise_multiplier, best_model_state).
            If use_dp is False and all_metrics is True:
                Returns (val_loss, accuracy, precision, recall, f1, best_model_state).
            If all_metrics is False:
                Returns best F1 score achieved.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Fall back to CPU when no GPU
    # print("🚀 Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
    print("Using device:", device)

    model.to(device)

    # Compute pos_weight for class imbalance handling
    pos_weight = torch.tensor([compute_class_balance(train_loader)], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if use_dp:
        privacy_engine = PrivacyEngine(secure_mode=SECURE_MODE)
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            criterion=criterion,
            data_loader=train_loader,
            target_epsilon=target_epsilon,
            target_delta=delta,
            epochs=epochs,
            max_grad_norm=clipping_norm
        )
        noise_multiplier = optimizer.noise_multiplier

        print(color_text(f"DP-SGD enabled: Target ε={target_epsilon}, δ={delta}, Clip={clipping_norm}, Noise Multiplier={noise_multiplier:.4f}", BRIGHT_BLUE))
    else: privacy_engine = None
    
    # Tracking variables for early stopping
    best_f1 = 0
    patience = EARLY_STOPPING_PATIENCE
    epochs_no_improve = 0
    metrics = {
        "val_loss": 0,
        "acc": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0
    }
    best_model_state = None

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Convert to (batch, channels, timesteps) format for Conv1D compatibility
            X_batch = X_batch.permute(0, 2, 1)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                X_batch = X_batch.permute(0, 2, 1)
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()
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
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        epoch_logging = print_timestamp_now() + color_text(f" Epoch {epoch+1}: ", BLUE)

        if use_dp and privacy_engine is not None:
            actual_epsilon = privacy_engine.get_epsilon(delta)
            epoch_logging += (
                f"Target ε = {target_epsilon:.4f},"
                f" Actual ε = {actual_epsilon:.4f},"
                f" δ = {delta}\n"
            )

        epoch_logging += (
            f"Training Loss = {avg_train_loss:.4f}, "
            f"Validation Loss = {avg_val_loss:.4f}, "
            f"Accuracy = {acc:.4f}, "
            f"Precision = {precision:.4f}, "
            f"Recall = {recall:.4f}, "
            f"F1 = {f1:.4f}"
        )

        print(epoch_logging)
        # Update scheduler based on validation F1 score
        scheduler.step(f1)

        # Early stopping logic
        if f1 > best_f1:
            best_f1 = f1
            epochs_no_improve = 0
            metrics["val_loss"] = avg_val_loss
            metrics["acc"] = acc
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1"] = f1
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break # Early stopping...
    
    if use_dp:
        return (metrics["val_loss"], metrics["acc"], metrics["precision"], metrics["recall"], metrics["f1"], privacy_engine.get_epsilon(delta), noise_multiplier, best_model_state) if all_metrics else best_f1
    else:
        return (metrics["val_loss"], metrics["acc"], metrics["precision"], metrics["recall"], metrics["f1"], best_model_state) if all_metrics else best_f1

def evaluate_model(model, val_loader):
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

def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning.

    Suggests values for learning rate, dropout, clipping norm, number of channels, and CNN layer count.
    Trains a DP-SGD CNN model using Opacus under a fixed ε value and evaluates performance on validation data.

    Logs training metrics and hyperparameters to CSV, saves the model if it's the best so far,
    and returns the F1 score as the optimization target.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object used to suggest and record hyperparameters.

    Returns:
        float: Validation F1 score achieved by the model with the current trial's hyperparameters.
    """
    global initial_completed, training_start_time

    learning_rate = trial.suggest_float("learning_rate", OPTUNA_LR_MIN, OPTUNA_LR_MAX, log=True)
    dropout = trial.suggest_float("dropout", OPTUNA_DROPOUT_MIN, OPTUNA_DROPOUT_MAX)
    clipping_norm = trial.suggest_float("clipping_norm", OPTUNA_CLIPPING_NORM_MIN, OPTUNA_CLIPPING_NORM_MAX)
    num_channels = trial.suggest_categorical("num_channels", OPTUNA_NUM_CHANNELS_VALS)
    layer_count = trial.suggest_int("layer_count", OPTUNA_LAYERS_MIN, OPTUNA_LAYERS_MAX)

    # Calculate how many trials are left in this training run
    trials_left = OPTUNA_N_TRIALS - (trial.number - initial_completed)

    # Log current trial hyperparameter details
    print('_____________________________________________________________')
    print(color_text(f"\nTrial {trial.number}: Testing DP-SGD CNN with ε = {EPSILON_VAL}...", RED))
    print(f"Learning Rate: {learning_rate}")
    print(f"Dropout: {dropout}")
    print(f"Clipping Norm: {clipping_norm}")
    print(f"Number of Channels: {num_channels}")
    print(f"Layer Count: {layer_count}")
    print(f"{trials_left}/{OPTUNA_N_TRIALS} trials remaining...\n")

    # Load data and train
    train_loader, val_loader = load_data(X_PATH, Y_PATH, DEFAULT_BATCH_SIZE)
    model = make_cnn(layer_count, num_channels, dropout)
    val_loss, accuracy, precision, recall, f1, actual_epsilon, noise_multiplier, opacus_model_state = train_model(
        model,
        train_loader,
        val_loader,
        learning_rate,
        use_dp=True,
        target_epsilon=EPSILON_VAL,
        delta=DELTA_VAL,
        clipping_norm=clipping_norm
    )
    
    trial.set_user_attr("actual_epsilon", actual_epsilon)
    trial.set_user_attr("noise_multiplier", noise_multiplier)
    trial.set_user_attr("f1", f1)
    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("precision", precision)
    trial.set_user_attr("recall", recall)
    trial.set_user_attr("val_loss", val_loss)
    trial.set_user_attr("batch_size", DEFAULT_BATCH_SIZE)

    write_headers = not os.path.exists(OPTUNA_RESULTS_FILE_PATH)

    with open(OPTUNA_RESULTS_FILE_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_headers:
            writer.writerow([
                "trial", "target-ε", "actual_ε", "noise_multiplier", "f1", "accuracy", "precision", "recall", "val_loss",
                "learning_rate", "dropout", "batch_size", "num_channels", "layer_count", "datetime"
            ])
        writer.writerow([
            trial.number, EPSILON_VAL, actual_epsilon, noise_multiplier, f1, accuracy, precision, recall, val_loss,
            learning_rate, dropout, DEFAULT_BATCH_SIZE, num_channels, layer_count, training_start_time
        ])

    if opacus_model_state:
        model_state = remove_opacus_prefix(opacus_model_state)
        
        os.makedirs(DP_MODEL_DIR_PATH, exist_ok=True)
        torch.save(model_state, DP_MODEL_DIR_PATH  / f"dp-model-t{trial.number}-f1{f1:.3f}-{training_start_time}.pt")

        if f1 > objective.best_f1:
            torch.save(model_state, BEST_MODEL_FILE_PATH)
            objective.best_f1 = f1

    return f1  # Maximizing F1 score

def main():
    """
    Runs Optuna-based hyperparameter tuning for a CNN fall detection model
    trained with differential privacy (DP-SGD) using Opacus.

    - Optimizes key parameters (e.g., learning rate, dropout, clipping norm) under a fixed ε.
    - Saves model weights, metrics, and best parameters to disk.
    - Evaluates the best DP model and logs performance to CSV for reproducibility.

    Returns:
        None
    """
    global initial_completed, training_start_time
    
    # === Testing Hyperparameter Combinations to Optimize DP Model Performance ===
    objective.best_f1 = 0.0 # Tracking best overall model

    os.makedirs(METRICS_DIR_PATH, exist_ok=True)
    os.makedirs(OPTUNA_STORAGE_DIR_PATH, exist_ok=True)

    print(color_text(f"{'🔒' if SECURE_MODE else '🔓'}Secure RNG Mode: {'ON' if SECURE_MODE else 'OFF'}", RED))

    study = optuna.create_study(
                        study_name=OPTUNA_STUDY_NAME,
                        direction="maximize",
                        storage=OPTUNA_STORAGE_PATH,
                        load_if_exists=True # Resume progress if exists
                    )
    initial_completed = len(study.trials)
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS)

    print(color_text(f"{OPTUNA_STUDY_NAME} Study Summary (ε = {EPSILON_VAL}):", BRIGHT_MAGENTA))
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print("Performance Metrics:")
    for attr_key, attr_value in study.best_trial.user_attrs.items():
        print(f"{attr_key}: {attr_value}")
    print(color_text(f"Best F1 Score: {study.best_trial.value:.4f}", RED))

    # === Best DP Model Evaluation ===
    print(color_text("\nEvaluating best DP model with best trial parameters...", PURPLE))

    best_params = study.best_trial.params
    best_model = make_cnn(
        layer_count=best_params["layer_count"],
        num_channels=best_params["num_channels"],
        dropout=best_params["dropout"]
    )
    best_model.load_state_dict(torch.load(BEST_MODEL_FILE_PATH, weights_only=True))

    # Reload data with best batch size
    _, val_loader = load_data(X_PATH, Y_PATH, DEFAULT_BATCH_SIZE)

    # Re-evaluate on validation data
    accuracy, precision, recall, f1 = evaluate_model(best_model, val_loader)

    print(color_text(f"\nFinal Evaluation of Best DP Model with ε = {EPSILON_VAL}:", BRIGHT_MAGENTA))
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    with open(BEST_MODEL_HYPERPARAMS_FILE_PATH, "w", newline="") as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(["learning_rate", "dropout", "clipping_norm", "num_channels", "layer_count", "batch_size", "datetime"])
        csvWriter.writerow(
            [best_params["learning_rate"],  
             best_params["dropout"],
             best_params["clipping_norm"],
             best_params["num_channels"],
             best_params["layer_count"],
             DEFAULT_BATCH_SIZE,
             training_start_time]
            )

    # Output FINAL best DP model performance metrics to CSV file
    with open(BEST_MODEL_METRICS_FILE_PATH, "w", newline="") as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(["target-ε", "f1", "accuracy", "precision", "recall", "datetime"])
        csvWriter.writerow([EPSILON_VAL, f1, accuracy, precision, recall, training_start_time])

if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    main()
