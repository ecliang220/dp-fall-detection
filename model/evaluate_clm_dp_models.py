"""
evaluate_clm_dp_models.py

Evaluates the performance of pre-trained fall detection and identity inference models 
on IMU data noised with the Correlated Laplace Mechanism for Differntial Privacy (CLM-DP) 
under varying privacy budgets (ε).

This script:
    - Loads the best-performing model weights and hyperparameters for both classifiers.
    - Iterates over a set of ε values corresponding to different levels of differential privacy.
    - Loads the corresponding noised input windows and dual labels.
    - Evaluates both models using standard classification metrics (accuracy, precision, recall, F1).
    - Logs all results to a CSV file for analysis of the privacy–utility tradeoff.

Inputs:
    - Model weight files (best checkpoints for both tasks)
    - Hyperparameter files (CSV)
    - CLM-DP noised sensor windows and dual labels

Outputs:
    - CSV file containing performance metrics for each ε level:
        `clm_dp_eval_metrics.csv`

Author: Ellie Liang
Date: 2025-06-09
"""
import csv
import sys
import os

from pathlib import Path
import torch

# Add project root to sys.path so `util` functions can be found
sys.path.append(str(Path(__file__).resolve().parents[1]))
from util.util import print_color_text_with_timestamp, get_timestamp_now
from model.model_util import (
    BEST_FALL_MODEL_FILE_PATH, BEST_IDENTITY_MODEL_FILE_PATH, 
    BEST_FALL_MODEL_HYPERPARAMS_FILE_PATH, BEST_IDENTITY_MODEL_HYPERPARAMS_FILE_PATH, 
    WINDOWS_DIR_PATH, WINDOWS_FILE_NAME, 
    FALL_LABELS_FILE_NAME, IDENTITY_LABELS_FILE_NAME,
    CLM_DP_METRICS_DIR_PATH, CLM_DP_EVAL_METRICS_FILE_PATH,
    set_seed, get_params_dict_from_csv, load_data,
    make_fall_detection_cnn, make_identity_inference_cnn,
    evaluate_fall_detection_model, evaluate_identity_inference_model,
    print_model_metrics,
    CLM_DP_Experiment
)

def main():
    """
    Evaluates fall detection and identity inference models on sensor data noised under 
    different CLM-DP ε levels.

    For each ε in the experiment:
        - Loads noised IMU data and corresponding fall and identity labels.
        - Applies the best-performing models to the noised input.
        - Computes and prints performance metrics: accuracy, precision, recall, and F1-score.
        - Logs results to a CSV file for utility–privacy analysis.

    This function assumes that models and noised data are already prepared and saved 
    in the appropriate directory structure.
    """
    # === Model Loading ===
    # Load hyperparams that yielded best performance metrics for FALL DETECTION MODEL
    fall_hyperparams = get_params_dict_from_csv(BEST_FALL_MODEL_HYPERPARAMS_FILE_PATH)
    best_fall_model = make_fall_detection_cnn(
        layer_count=fall_hyperparams["layer_count"],
        num_channels=fall_hyperparams["num_channels"],
        dropout=fall_hyperparams["dropout"]
    )
    state_dict = torch.load(BEST_FALL_MODEL_FILE_PATH, weights_only=True)
    best_fall_model.load_state_dict(state_dict)

    # Load hyperparams that yielded best performance metrics for IDENTITY INFERENCE MODEL
    identity_hyperparams = get_params_dict_from_csv(BEST_IDENTITY_MODEL_HYPERPARAMS_FILE_PATH)
    best_identity_model = make_identity_inference_cnn(
        layer_count=identity_hyperparams["layer_count"],
        num_channels=identity_hyperparams["num_channels"],
        dropout=identity_hyperparams["dropout"]
    )
    state_dict = torch.load(BEST_IDENTITY_MODEL_FILE_PATH, weights_only=True)
    best_identity_model.load_state_dict(state_dict)

    os.makedirs(CLM_DP_METRICS_DIR_PATH, exist_ok=True)
    if not os.path.exists(CLM_DP_EVAL_METRICS_FILE_PATH):
        with open(CLM_DP_EVAL_METRICS_FILE_PATH, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "epsilon",
                "fall_f1", "fall_accuracy", "fall_precision", "fall_recall",
                "identity_f1", "identity_accuracy", "identity_precision", "identity_recall",
                "timestamp"
            ])

    for epsilon in CLM_DP_Experiment.EPSILON_VALS:
        # === Data Loading ===
        clm_dp_output_dir_path = WINDOWS_DIR_PATH / f"clm_dp_eps_{epsilon}"
        X_path = clm_dp_output_dir_path / WINDOWS_FILE_NAME
        y_fall_path = clm_dp_output_dir_path / FALL_LABELS_FILE_NAME
        y_identity_path = clm_dp_output_dir_path / IDENTITY_LABELS_FILE_NAME

        _, fall_val_loader = load_data(X_path, y_fall_path, fall_hyperparams["batch_size"])
        _, identity_val_loader = load_data(X_path, y_identity_path, identity_hyperparams["batch_size"])

        # === Metrics Evaluation ===
        print_color_text_with_timestamp(f"Evaluating Models under CLM-DP with ε = {epsilon}...", "PURPLE")
        fall_accuracy, fall_precision, fall_recall, fall_f1 = evaluate_fall_detection_model(best_fall_model, fall_val_loader)
        print_model_metrics(fall_accuracy, fall_precision, fall_recall, fall_f1, summary_title=f"Evaluation of Fall Detection (ε = {epsilon})")
        
        identity_accuracy, identity_precision, identity_recall, identity_f1 = evaluate_identity_inference_model(best_identity_model, identity_val_loader)
        print_model_metrics(identity_accuracy, identity_precision, identity_recall, identity_f1, summary_title=f"Evaluation of Identity Inference (ε = {epsilon})")
        
        # === CSV Logging ===
        with open(CLM_DP_EVAL_METRICS_FILE_PATH, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                f"{epsilon:.2f}", 
                fall_f1, fall_accuracy, fall_precision, fall_recall,
                identity_f1, identity_accuracy, identity_precision, identity_recall,
                get_timestamp_now()
                ])

if __name__ == '__main__':
    set_seed() # Default seed value is 42
    main()