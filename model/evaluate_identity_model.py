import os
import csv
import sys
from pathlib import Path

import optuna
import torch

# Add project root to sys.path so `util` functions can be found
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.model_util import *
from util.util import print_with_timestamp, print_color_text

def main():
    print_with_timestamp("\nEvaluating best model with best trial parameters...")

    if BEST_IDENTITY_MODEL_HYPERPARAMS_FILE_PATH.exists():
        # Load best hyperparams from CSV file 
        identity_hyperparams = dict()
        with open(BEST_IDENTITY_MODEL_HYPERPARAMS_FILE_PATH, 'r') as csv_file:
            dict_reader = csv.DictReader(csv_file)
            identity_hyperparams = next(dict_reader, {})
            identity_hyperparams = {
                "layer_count": int(identity_hyperparams["layer_count"]),
                "learning_rate": float(identity_hyperparams["learning_rate"]),
                "batch_size": int(identity_hyperparams["batch_size"]),
                "num_channels": int(identity_hyperparams["num_channels"]),
                "dropout": float(identity_hyperparams["dropout"]),
            }
    elif os.path.exists(OPTUNA_IDENTITY_STORAGE_PATH): # Load from existing Optuna study
        study = optuna.create_study(
                            study_name=OPTUNA_IDENTITY_STUDY_NAME,
                            direction='maximize',
                            storage=OPTUNA_IDENTITY_STORAGE_PATH,
                            load_if_exists=True
                        )
        identity_hyperparams = study.best_trial.params
    else:
        print("Error: cannot reload best hyperparameters for identity inference model")
        print("Unable to build CNN for evaluation without hyperparameters. Exiting script...")
        return
    
    best_model = make_identity_inference_cnn(
        layer_count=identity_hyperparams["layer_count"],
        num_channels=identity_hyperparams["num_channels"],
        dropout=identity_hyperparams["dropout"]
    )
    best_model.load_state_dict(torch.load(BEST_IDENTITY_MODEL_FILE_PATH, weights_only=True))

    # Reload data with best batch size
    _, val_loader = load_data(X_PATH, Y_IDENTITY_PATH, identity_hyperparams["batch_size"])

    # Re-evaluate on validation data
    acc, precision, recall, f1 = evaluate_identity_inference_model(best_model, val_loader)

    print("\nFinal Evaluation of Best Model:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print_with_timestamp(f"Saving best hyperparams...")
    print_color_text(str(BEST_IDENTITY_MODEL_HYPERPARAMS_FILE_PATH), "BLUE")
    with open(BEST_IDENTITY_MODEL_HYPERPARAMS_FILE_PATH, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["layer_count", "learning_rate", "batch_size", "num_channels", "dropout"])
        csv_writer.writerow(
            [identity_hyperparams["layer_count"], 
             identity_hyperparams["lr"], 
             identity_hyperparams["batch_size"], 
             identity_hyperparams["num_channels"], 
             identity_hyperparams["dropout"]]
            )
    
    os.makedirs(IDENTITY_METRICS_DIR_PATH, exist_ok=True)

    print_with_timestamp(f"Saving best performance metrics...")
    print_color_text(str(BEST_IDENTITY_MODEL_METRICS_FILE_PATH), "BLUE")
    # Output FINAL best model performance metrics to CSV file
    with open(BEST_IDENTITY_MODEL_METRICS_FILE_PATH, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["accuracy", "precision", "recall", "f1"])
        csv_writer.writerow([acc, precision, recall, f1])


if __name__ == '__main__':
    set_seed() # Default to 42
    main()