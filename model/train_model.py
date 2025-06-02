import csv
import os
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
import optuna

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

# Random seed for reproducibility across dataset splits, model initialization, and Optuna trials
RANDOM_SEED = 42

# Number of epochs to train the CNN model during each trial
NUM_EPOCHS = 30

# Threshold applied to sigmoid output to determine binary class (1 if output > threshold, else 0)
SIGMOID_BINARY_CLASSIFICATION_THRESHOLD = 0.5

"""
Optuna Config
"""
OPTUNA_STORAGE_DIR_PATH = PROJECT_ROOT / "storage"
OPTUNA_STORAGE_PATH = f"sqlite:///{str(Path(PROJECT_ROOT) / 'storage' / 'optuna_fall_detection.db')}"
OPTUNA_RESULTS_FILE_PATH = METRICS_DIR_PATH / "optuna_results.csv"

OPTUNA_STUDY_NAME = "CNN_fall_detection_optimization"
OPTUNA_N_TRIALS = 30
OPTUNA_LR_MIN = 1e-4
OPTUNA_LR_MAX = 1e-2
OPTUNA_DROPOUT_MIN = 0.3
OPTUNA_DROPOUT_MAX = 0.6
OPTUNA_BATCH_SIZE_VALS = [16, 32, 64, 128]
OPTUNA_NUM_CHANNELS_VALS = [64, 128, 256]
OPTUNA_LAYERS_MIN = 3
OPTUNA_LAYERS_MAX = 5

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

def train_model(model, train_loader, val_loader, learning_rate, epochs=NUM_EPOCHS, all_metrics=True):
    """
    Trains and evaluates the CNN model on fall detection data.

    - Uses BCEWithLogitsLoss with class imbalance correction via pos_weight.
    - Applies Adam optimizer and learning rate scheduling based on validation F1 score.
    - Performs early stopping if no F1 improvement is observed for `patience` epochs.
    - Tracks training and validation performance, including accuracy, precision, recall, and F1.

    Args:
        model (nn.Module): The CNN model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        learning_rate (float): Learning rate for the Adam optimizer.
        epochs (int, optional): Maximum number of training epochs. Defaults to NUM_EPOCHS.
        all_metrics (bool, optional): Whether to return all metrics and model state. If False, only F1 is returned.

    Returns:
        tuple or float:
            If all_metrics is True, returns (val_loss, acc, precision, recall, f1, best_model_state).
            Otherwise, returns only the best F1 score achieved.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Fall back to CPU when no GPU
    print("🚀 Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
    print("Using device:", device)

    model.to(device)

    # Compute pos_weight for class imbalance handling
    pos_weight = torch.tensor([compute_class_balance(train_loader)], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Tracking variables for early stopping
    best_f1 = 0
    patience = 5
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

        print(f"Epoch {epoch+1}: Training Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}, Accuracy = {acc:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")

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
    
    return (metrics["val_loss"], metrics["acc"], metrics["precision"], metrics["recall"], metrics["f1"], best_model_state) if all_metrics else best_f1

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

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.

    - Samples a set of hyperparameters from defined search spaces.
    - Constructs and trains a CNN model with the suggested parameters.
    - Evaluates model performance on the validation set.
    - Saves the current trial's performance metrics and hyperparameters to a CSV file.
    - Saves the model checkpoint if it achieves the best F1 score so far.

    Args:
        trial (optuna.trial.Trial): A trial object from Optuna used to suggest hyperparameters 
                                    and store results.

    Returns:
        float: The F1 score on the validation set, used as the optimization objective.
    """
    # Hyperparameters to explore
    lr = trial.suggest_float("lr", OPTUNA_LR_MIN, OPTUNA_LR_MAX, log=True)
    dropout = trial.suggest_float("dropout", OPTUNA_DROPOUT_MIN, OPTUNA_DROPOUT_MAX)
    batch_size = trial.suggest_categorical("batch_size", OPTUNA_BATCH_SIZE_VALS)
    num_channels = trial.suggest_categorical("num_channels", OPTUNA_NUM_CHANNELS_VALS)
    layer_count = trial.suggest_int("layer_count", OPTUNA_LAYERS_MIN, OPTUNA_LAYERS_MAX)

    # Log current trial hyperparameter details
    print(f"\nTrial {trial}: Testing {layer_count}-layer CNN...")
    print(f"Learning Rate: {lr}")
    print(f"Batch Size: {batch_size}")
    print(f"Number of Channels: {num_channels}")
    print(f"Dropout: {dropout}")

    # Load data and train
    train_loader, val_loader = load_data(X_PATH, Y_PATH, batch_size)
    model = make_cnn(layer_count, num_channels, dropout)
    val_loss, acc, precision, recall, f1, model_state = train_model(model, train_loader, val_loader, lr)

    trial.set_user_attr("accuracy", acc)
    trial.set_user_attr("precision", precision)
    trial.set_user_attr("recall", recall)
    trial.set_user_attr("val_loss", val_loss)

    write_headers = not os.path.exists(OPTUNA_RESULTS_FILE_PATH)

    with open(OPTUNA_RESULTS_FILE_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_headers:
            writer.writerow([
                "trial", "f1", "precision", "recall", "accuracy", "val_loss",
                "learning_rate", "dropout", "batch_size", "num_channels", "layer_count"
            ])
        writer.writerow([
            trial.number, f1, precision, recall, acc, val_loss,
            lr, dropout, batch_size, num_channels, layer_count
        ])

    if model_state:
        os.makedirs(MODEL_DIR_PATH, exist_ok=True)
        torch.save(model_state, MODEL_DIR_PATH  / f"model-t{trial.number}-lc{layer_count}-f1{f1:.3f}.pt")

        if f1 > objective.best_f1:
            torch.save(model_state, BEST_MODEL_FILE_PATH)
            objective.best_f1 = f1

    return f1  # Maximizing F1 score

def main():
    """
    Runs Optuna hyperparameter optimization and evaluates the best CNN model.

    - Creates or resumes an Optuna study to tune CNN architecture and training parameters.
    - Trains and evaluates models across multiple trials, logging results.
    - Saves metrics and model checkpoints for each trial.
    - Loads and evaluates the best-performing model on the validation dataset.
    - Saves final best model performance metrics to a CSV file.

    Returns:
        None
    """
    # === Testing Hyperparameter Combinations to Optimize Model Performance  ===
    objective.best_f1 = 0.0 # Tracking best overall model

    os.makedirs(METRICS_DIR_PATH, exist_ok=True)
    os.makedirs(OPTUNA_STORAGE_DIR_PATH, exist_ok=True)
    if os.path.exists(OPTUNA_RESULTS_FILE_PATH): os.remove(OPTUNA_RESULTS_FILE_PATH)

    study = optuna.create_study(
                        study_name=OPTUNA_STUDY_NAME,
                        direction="maximize",
                        storage=OPTUNA_STORAGE_PATH,
                        load_if_exists=True # Resume progress if exists
                    )
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS)

    print("Number of finished trials:", len(study.trials))
    print("\nBest trial:")
    print(study.best_trial)
    print("Performance Metrics:")
    print(study.best_trial.user_attrs)
    print(f"Best F1 Score: {study.best_trial.value:.4f}")

    # === Best Model Evaluation ===
    print("\nEvaluating best model with best trial parameters...")

    best_params = study.best_trial.params
    best_model = make_cnn(
        layer_count=best_params["layer_count"],
        num_channels=best_params["num_channels"],
        dropout=best_params["dropout"]
    )
    best_model.load_state_dict(torch.load(BEST_MODEL_FILE_PATH))

    # Reload data with best batch size
    _, val_loader = load_data(X_PATH, Y_PATH, best_params["batch_size"])

    # Re-evaluate on validation data
    acc, precision, recall, f1 = evaluate_model(best_model, val_loader)

    print("\nFinal Evaluation of Best Model:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Output FINAL best model performance metrics to CSV file
    with open(BEST_MODEL_METRICS_FILE_PATH, "w", newline="") as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(["accuracy", "precision", "recall", "f1"])
        csvWriter.writerow([acc, precision, recall, f1])

    # TODO: Uncomment for running model without tuning hyperparameteres
    # # Load Data
    # train_loader, val_loader = load_data(X_PATH, Y_PATH, BATCH_SIZE)

    # # Testing n-layer CNN
    # print(f"\nTesting {LAYER_COUNT}-layer CNN...")
    # print(f"Learning Rate: {LEARNING_RATE}")
    # print(f"Batch Size: {BATCH_SIZE}")
    # print(f"Number of Channels: {NUM_CHANNELS}")
    # # print(f"Number of Epochs: {NUM_EPOCHS}")
    # print(f"Dropout: {DROPOUT}")
    # print()
    # model = make_cnn(LAYER_COUNT, NUM_CHANNELS, DROPOUT)
    # train_model(model, train_loader, val_loader, LEARNING_RATE)

    # # Testing 3-layer CNN
    # print("\nTesting 3-layer CNN...")
    # model_3 = make_cnn(layer_count=3)
    # train_model(model_3, train_loader, val_loader)

    # # Testing 5-layer CNN
    # print("\nTesting 5-layer CNN...")
    # model_5 = make_cnn(layer_count=5)
    # train_model(model_5, train_loader, val_loader)

    # # Testing 7-layer CNN
    # print("\nTesting 7-layer CNN...")
    # model_7 = make_cnn(layer_count=7)
    # train_model(model_7, train_loader, val_loader)

    # # Testing 10-layer CNN
    # print("\nTesting 10-layer CNN...")
    # model_10 = make_cnn(layer_count=10)
    # train_model(model_10, train_loader, val_loader)

if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    main()
