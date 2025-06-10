import os
import csv
import sys
from pathlib import Path

import optuna
import torch

# Add project root to sys.path so `util` functions can be found
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.model_util import *
from util.util import print_with_timestamp, print_color_text_with_timestamp, print_color_text

def main():
    print_with_timestamp("\nEvaluating best model with best trial parameters...")

    study = optuna.create_study(
                        study_name=OPTUNA_IDENTITY_STUDY_NAME,
                        direction='maximize',
                        storage=OPTUNA_IDENTITY_STORAGE_PATH,
                        load_if_exists=True
                    )

    best_params = study.best_trial.params
    best_model = make_identity_inference_cnn(
        layer_count=best_params["layer_count"],
        num_channels=best_params["num_channels"],
        dropout=best_params["dropout"]
    )
    best_model.load_state_dict(torch.load(BEST_IDENTITY_MODEL_FILE_PATH, weights_only=True))

    # Reload data with best batch size
    _, val_loader = load_data(X_PATH, Y_IDENTITY_PATH, best_params["batch_size"])

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
            [best_params["layer_count"], 
             best_params["lr"], 
             best_params["batch_size"], 
             best_params["num_channels"], 
             best_params["dropout"]]
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