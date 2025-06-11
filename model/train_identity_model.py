import csv
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path

# Add project root to sys.path so `util` functions can be found
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.model_util import *
from util.util import print_color_text_with_timestamp, print_color_text, print_with_timestamp, bold_text, color_text

# Initial number of trials completed when Optuna study is loaded
initial_completed = 0

def train_model(model, train_loader, val_loader, learning_rate, epochs=NUM_EPOCHS, all_metrics=True):
    """
    Trains and evaluates the CNN model on the identity inference dataset.

    - Uses CrossEntropyLoss for multi-class classification.
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
    # print("🚀 Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
    print("Using device:", device)

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
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

                preds = torch.argmax(output, dim=1)
                correct += (preds == y_batch.view(-1)).sum().item()
                total += y_batch.size(0)

                all_preds.append(preds.cpu()) 
                all_targets.append(y_batch.cpu())

        acc = correct / total

        y_true = torch.cat(all_targets).numpy()
        y_pred = torch.cat(all_preds).numpy()

        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0) 
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0) 
        f1 = f1_score(y_true, y_pred, average="weighted")  
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print_with_timestamp(
            f"{color_text(bold_text(f'Epoch {epoch+1}:'), 'BLUE')} Training Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}, Accuracy = {acc:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}"
        )

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

def main():
    """
    Trains identity inference model using pre-tuned hyperparams, and evaluates model based on standard
    performance metrics (F1, Accuracy, Precision, Recall)

    Returns:
        None
    """
    # === Training Model ===
    if BEST_IDENTITY_MODEL_HYPERPARAMS_FILE_PATH.exists():
        # Load best hyperparams from CSV file 
        identity_hyperparams = dict()
        with open(BEST_IDENTITY_MODEL_HYPERPARAMS_FILE_PATH, 'r') as csv_file:
            dict_reader = csv.DictReader(csv_file)
            identity_hyperparams = next(dict_reader, {})
            identity_hyperparams = {
                "layer_count": int(identity_hyperparams.get("layer_count", DEFAULT_LAYER_COUNT)),
                "learning_rate": float(identity_hyperparams.get("learning_rate", DEFAULT_LEARNING_RATE)),
                "batch_size": int(identity_hyperparams.get("batch_size", DEFAULT_BATCH_SIZE)),
                "num_channels": int(identity_hyperparams.get("num_channels", DEFAULT_NUM_CHANNELS)),
                "dropout": float(identity_hyperparams.get("dropout", DEFAULT_DROPOUT)),
            }
        if not identity_hyperparams:
            print("Error: CSV does not contain valid hyperparameter values.")
            return
    else:
        print("Error: cannot reload best hyperparameters for identity inference model")
        print("Unable to build CNN for evaluation without hyperparameters. Exiting script...")
        return
    lc = identity_hyperparams.get("layer_count")
    lr = identity_hyperparams.get("learning_rate")
    bs = identity_hyperparams.get("batch_size")
    nc = identity_hyperparams.get("num_channels")
    dr = identity_hyperparams.get("dropout")
    
    model = make_identity_inference_cnn(
        layer_count=lc,
        num_channels=nc,
        dropout=dr
    )

    print_color_text_with_timestamp(f"Training {lc}-layer Identity CNN...", "BRIGHT_MAGENTA")
    print(f"Learning Rate: {lr}")
    print(f"Batch Size: {bs}")
    print(f"Number of Channels: {nc}")
    print(f"Dropout: {dr}")
    print()
    
    # Load data and train
    train_loader, val_loader = load_data(X_PATH, Y_IDENTITY_PATH, bs)
    # (metrics["val_loss"], metrics["acc"], metrics["precision"], metrics["recall"], metrics["f1"], best_model_state)
    _, accuracy, precision, recall, f1, model_state = train_model(model, train_loader, val_loader, lr, epochs=NUM_EPOCHS, all_metrics=True)

    saved_model_file_name = f"model-lc{lc}-f1{f1:.3f}-acc{accuracy:.3f}.pt"

    if model_state:
        os.makedirs(IDENTITY_MODEL_DIR_PATH, exist_ok=True)
        torch.save(model_state, IDENTITY_MODEL_DIR_PATH  / saved_model_file_name)

    print_model_metrics(accuracy, precision, recall, f1, summary_title="Trained Identity Model Results")

    # === Evaluating Model ===
    print_color_text("Evaluating trained identity inference model...", "PURPLE")

    best_model = make_identity_inference_cnn(
        layer_count=identity_hyperparams.get("layer_count", DEFAULT_LAYER_COUNT),
        num_channels=identity_hyperparams.get("num_channels", DEFAULT_NUM_CHANNELS),
        dropout=identity_hyperparams.get("dropout", DEFAULT_DROPOUT)
    )
    best_model.load_state_dict(torch.load(IDENTITY_MODEL_DIR_PATH / saved_model_file_name, weights_only=True))


    # Reload data with best batch size
    _, val_loader = load_data(X_PATH, Y_IDENTITY_PATH, bs)

    # Re-evaluate on validation data
    accuracy, precision, recall, f1 = evaluate_identity_inference_model(best_model, val_loader)

    print_model_metrics(accuracy, precision, recall, f1, summary_title="Identity Inference Model Evaluation")

    os.makedirs(IDENTITY_METRICS_DIR_PATH, exist_ok=True)
    print_with_timestamp(f"Saving best performance metrics...")
    print_color_text(str(BEST_IDENTITY_MODEL_METRICS_FILE_PATH), "BLUE")
    # Output FINAL best model performance metrics to CSV file
    with open(BEST_IDENTITY_MODEL_METRICS_FILE_PATH, "w", newline="") as csv_file:
        csvWriter = csv.writer(csv_file)
        csvWriter.writerow(["accuracy", "precision", "recall", "f1"])
        csvWriter.writerow([accuracy, precision, recall, f1])

if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    main()
