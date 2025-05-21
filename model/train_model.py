import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score  # TODO: review
from pathlib import Path

# Training model input files: windows and labels as NumPy arrays in compressed binary format (.npy)
X_path = Path("../data/windows/X_windows.npy")
y_path = Path("../data/windows/y_labels.npy")

def flatten_and_normalize_data(X):
    """
    Normalizes the input sensor data across all time steps and channels.

    Args:
        X (np.ndarray): A 3D NumPy array of shape (num_windows, time_steps, num_features).

    Returns:
        np.ndarray: A normalized array of the same shape with zero mean and unit variance per feature.
    """
    X_flat = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler().fit(X_flat)

    return scaler.transform(X_flat).reshape(X.shape)

def load_data(X_path, y_path):
    """
    Loads and prepares data for training and validation.

    - Loads windowed sensor data and labels from `.npy` files.
    - Normalizes the input features using `StandardScaler`.
    - Converts the arrays to PyTorch tensors.
    - Splits into training and validation datasets (80/20 split).

    Args:
        X_path (Path): Path to the input windowed feature array (.npy).
        y_path (Path): Path to the corresponding label array (.npy).

    Returns:
        tuple: DataLoader objects for training and validation.
    """
    # Load data as NumPy arrays
    X = np.load(X_path)
    y = np.load(y_path)

    X_scaled = flatten_and_normalize_data(X)

    # Convert to torch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)

    return train_loader, val_loader

def make_cnn(layer_count):
    """
    Base CNN model: Builds a 1D CNN model with a variable number of layers.
    Add more convolutional layers based on desired depth

    - All models begin with a Conv1D + ReLU + MaxPool layer.
    - Base case: 3-layer model is the minimum number of layers tested
    - 5-layer and 7-layer options add additional convolution blocks.
    - Output features are pooled and passed to a fully connected binary classifier.

    Args:
        layer_count (int): Number of convolutional layers (e.g., 3, 5, 7).

    Returns:
        nn.Sequential: A PyTorch model ready for training.
    """
    layers = [nn.Conv1d(9, 64, kernel_size=5), nn.ReLU(), nn.MaxPool1d(2)]
    if layer_count >= 5:
        layers += [nn.Conv1d(64, 128, kernel_size=3), nn.ReLU(), nn.MaxPool1d(2)]
    if layer_count >= 7:
        layers += [nn.Conv1d(128, 256, kernel_size=3), nn.ReLU(), nn.AdaptiveMaxPool1d(1)]
    else:
        layers += [nn.AdaptiveAvgPool1d(1)]

    model = nn.Sequential(
        *layers,
        nn.Flatten(),
        nn.Linear(256 if layer_count >= 7 else 128 if layer_count >= 5 else 64, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    return model

def train_model(model, train_loader, val_loader, epochs=20):
    """
    Trains and evaluates the CNN model on fall detection data.

    - Uses Binary Cross-Entropy loss for binary classification.
    - Monitors validation loss and accuracy per epoch.
    - Converts input to the (batch_size, channels, time_steps) format for Conv1D.

    Args:
        model (nn.Module): The CNN model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int): Number of training epochs to run.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Fall back to CPU when no GPU
    print("Using device:", device)
    
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Convert to (batch, channels, timesteps) format for Conv1D compatibility
            X_batch = X_batch.permute(0, 2, 1)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []  # TODO: review
        all_targets = []  # TODO: review

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                X_batch = X_batch.permute(0, 2, 1)
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()
                preds = (output > 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

                all_preds.append(preds.cpu())  # TODO: review
                all_targets.append(y_batch.cpu())  # TODO: review

        acc = correct / total

        y_true = torch.cat(all_targets).numpy()  # TODO: review
        y_pred = torch.cat(all_preds).numpy()  # TODO: review

        precision = precision_score(y_true, y_pred)  
        recall = recall_score(y_true, y_pred) 
        f1 = f1_score(y_true, y_pred)  

        print(f"Epoch {epoch+1}: Validation Loss = {val_loss:.4f}, Accuracy = {acc:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")

def main():
    """
    Loads data and trains three CNN models of increasing depth.

    - Compares 3-layer, 5-layer, and 7-layer CNNs on validation accuracy.
    - Intended for architecture comparison and benchmarking.

    Returns:
        None
    """
    # Load Data
    train_loader, val_loader = load_data(X_path, y_path)

    # Testing 3-layer CNN
    print("\nTesting 3-layer CNN...")
    model_3 = make_cnn(layer_count=3)
    train_model(model_3, train_loader, val_loader)

    # Testing 5-layer CNN
    print("\nTesting 5-layer CNN...")
    model_5 = make_cnn(layer_count=5)
    train_model(model_5, train_loader, val_loader)

    # Testing 7-layer CNN
    print("\nTesting 7-layer CNN...")
    model_7 = make_cnn(layer_count=7)
    train_model(model_7, train_loader, val_loader)

if __name__ == "__main__":
    main()
