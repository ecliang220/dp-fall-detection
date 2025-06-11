"""
evaluate_identity_model.py

Evaluates the best-performing identity inference CNN model on a validation dataset using pre-saved
hyperparameters and pre-trained weights. 

This script:
    - Loads model hyperparameters from CSV or Optuna study.
    - Loads trained model weights from disk.
    - Reloads preprocessed IMU windows and identity labels.
    - Evaluates model performance using accuracy, precision, recall, and F1-score.
    - Logs the performance metrics and hyperparameters to CSV for reproducibility.

Inputs:
    - Trained model weights
    - Hyperparameters (CSV or Optuna study)
    - IMU windowed dataset with identity labels

Outputs:
    - `best_identity_model_metrics.csv`: Performance metrics of the evaluated model
    - `best_identity_model_hyperparams.csv`: Hyperparameters used by the evaluated model

This script supports both standard and temporary override paths to facilitate validation
of new input formats against previously optimized models.

Author: Ellie Liang
Date: 2025-06-09
"""
import os
import csv
import sys
from pathlib import Path

import optuna
import torch

# Add project root to sys.path so `util` functions can be found
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.model_util import *
from util.util import print_with_timestamp, print_color_text, print_color_text_with_timestamp

def main():
    """
    Loads and evaluates the best identity inference model using pre-tuned hyperparameters.

    This function:
        - Loads hyperparameters from a CSV file or falls back to Optuna study if not available.
        - Loads the corresponding saved model weights from prior training.
        - Loads preprocessed IMU input windows and identity labels.
        - Evaluates the model on the validation set using classification metrics.
        - Prints and logs the model’s performance and associated hyperparameters to CSV files.
    """
    print_color_text_with_timestamp("Evaluating best model with best trial parameters...", "PURPLE")

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
        layer_count=identity_hyperparams.get("layer_count", DEFAULT_LAYER_COUNT),
        num_channels=identity_hyperparams.get("num_channels", DEFAULT_NUM_CHANNELS),
        dropout=identity_hyperparams.get("dropout", DEFAULT_DROPOUT)
    )

    best_model.load_state_dict(torch.load(BEST_IDENTITY_MODEL_FILE_PATH, weights_only=True))

    # Reload data with best batch size
    _, val_loader = load_data(X_PATH, Y_IDENTITY_PATH, identity_hyperparams.get("batch_size", DEFAULT_BATCH_SIZE))

    # Re-evaluate on validation data
    acc, precision, recall, f1 = evaluate_identity_inference_model(best_model, val_loader)

    print_model_metrics(acc, precision, recall, f1, summary_title="Evaluation of Best Identity Model")

    print_with_timestamp(f"Saving best hyperparams...")
    print_color_text(str(BEST_IDENTITY_MODEL_HYPERPARAMS_FILE_PATH), "BLUE")
    with open(BEST_IDENTITY_MODEL_HYPERPARAMS_FILE_PATH, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["layer_count", "learning_rate", "batch_size", "num_channels", "dropout"])
        csv_writer.writerow(
            [identity_hyperparams.get("layer_count", DEFAULT_LAYER_COUNT), 
             identity_hyperparams.get("learning_rate", DEFAULT_LEARNING_RATE), 
             identity_hyperparams.get("batch_size", DEFAULT_BATCH_SIZE), 
             identity_hyperparams.get("num_channels", DEFAULT_NUM_CHANNELS), 
             identity_hyperparams.get("dropout", DEFAULT_DROPOUT)]
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