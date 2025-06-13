"""
inject_clm_dp_noise.py

Applies temporally correlated Laplace noise to fixed-length IMU windows under 
Local Differential Privacy (LDP) using the Correlated Laplace Mechanism (CLM).
This enables evaluation of the privacy–utility tradeoff for fall detection and 
identity inference models.

Inputs:
    - X_windows.npy: Clean IMU windows (shape: [n, 200, 9])
    - y_labels.npy: Corresponding fall labels (binary)

Outputs:
    - clm_dp_eps_{ε}/X_windows.npy: Noised IMU windows at ε = {ε}
    - clm_dp_eps_{ε}/y_labels.npy: Unchanged labels (copied for consistency)

Author: Ellie Liang
Date: 2025-06-05
"""
import csv
import sys
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add project root to sys.path so `util` functions can be found
sys.path.append(str(Path(__file__).resolve().parents[1]))
from util.util import print_with_timestamp, print_color_text_with_timestamp, print_color_text, bold_text, get_timestamp_now

# --------------------------------------------------------------------
# Directory and File Paths
# --------------------------------------------------------------------
from model.model_util import (
    X_PATH, Y_FALL_PATH, Y_IDENTITY_PATH, 
    WINDOWS_DIR_PATH, WINDOWS_FILE_NAME, 
    FALL_LABELS_FILE_NAME, IDENTITY_LABELS_FILE_NAME, 
    LOG_FILE_PATH,
    WINDOW_SIZE, NUM_AXES, NUM_SENSORS, SENSOR_RANGE_IDX,
    CLM_DP_Experiment, # CLM-DP Experiment Constants
    set_seed
    )

# Boolean value indicating whether to skip or overwrite existing files for reruns of epsilon values
SKIP_EXISTING = False

def flatten_windows_to_2d(X_windows):
    """
    Flattens each 3D batch of sensor windows into a 2D array.

    Args:
        X_windows (np.ndarray): A 3D NumPy array of shape (num_windows, time_steps, num_features),
                                representing fixed-length IMU data windows.

    Returns:
        np.ndarray: A 2D array of shape (num_windows, time_steps × num_features), where each row
                    is a flattened version of one window.
    """
    return X_windows.reshape(X_windows.shape[0], -1)

def sample_correlated_laplace_noise(L, epsilon):
    """
    Generates temporally correlated multivariate Laplace noise using the Gaussian trick.

    Steps to employ Gaussian trick to construct a noise vector:
        1. Sample a multivariate normal vector z ~ N(0, Σ)  # Σ is a covariance matrix encoding temporal correlation via exponential decay
        2. Sample a scalar v ~ Exponential(1)
        3. Return z / sqrt(v), scaled by 1/ε to enforce the specified privacy budget

    Args:
        L (np.ndarray): Cholesky decomposition of the covariance matrix Σ.
        epsilon (float): Privacy budget (larger ε = less noise)

    Returns:
        np.ndarray: 1D array of correlated Laplace noise.
    """
    # Step 1: Sample z ~ N(0, Σ) with temporal correlation
    z = L @ np.random.randn(L.shape[0])

    # Step 2: Sample v ~ Exponential(1)
    v = np.random.exponential(scale=1.0)

    # Step 3: Apply Gaussian trick and scale for ε
    return (z / np.sqrt(v)) / epsilon

def inject_noise_into_windows(windows, epsilon, L):
    """
    Injects temporally correlated Laplace noise into flattened IMU windows.

    Args:
        windows (np.ndarray): Original IMU windows of shape (n, 200, 9).
        epsilon (float): Privacy budget for differential privacy.
        L (np.ndarray): Cholesky decomposition of the correlation matrix.

    Returns:
        np.ndarray: Noised IMU windows of shape (n, 1800).
    """
    X_flat = flatten_windows_to_2d(windows)  # shape: (num_windows, time_steps x features)

    X_noised = np.zeros_like(X_flat)

    print_color_text_with_timestamp(f"Injecting CLM DP noise at {bold_text(f'ε = {epsilon}')} into windows...", "BRIGHT_BLUE")
    
    for i in tqdm(range(X_flat.shape[0]), desc="Injecting noise", unit="window"): # Use progress bar
        # Inject CLM noise into each window
        noise = sample_correlated_laplace_noise(L, epsilon=epsilon)
        X_noised[i] = X_flat[i] + noise
    
    return X_noised

def precompute_window_cholesky_decomp(size):
    """
    Constructs and decomposes the exponential decay covariance matrix.

    Args:
        size (int): Dimensionality of the flattened window.

    Returns:
        np.ndarray: Lower triangular matrix from Cholesky decomposition.
    """
    # Precompute Cholesky decomposition of exponential decay covariance matrix
    indices = np.arange(size)
    corr_matrix = np.power(CLM_DP_Experiment.DECAY_FACTOR, np.abs(indices[:, None] - indices[None, :]))

    return np.linalg.cholesky(corr_matrix + CLM_DP_Experiment.JITTER_EPS * np.eye(size))

def precompute_sensor_cholesky_decomp(num_timesteps, num_axes):
    """
    Constructs and decomposes the exponential decay covariance matrix.

    Args:
        num_timesteps (int): Number of timesteps collected by sensor in a window.
        num_axes (int): Number of axes measured by sensor

    Returns:
        np.ndarray: Lower triangular matrix from Cholesky decomposition.
    """
    # Precompute Cholesky decomposition of exponential decay covariance matrix
    indices = np.arange(num_timesteps * num_axes)
    corr_matrix = np.power(CLM_DP_Experiment.DECAY_FACTOR, np.abs(indices[:, None] - indices[None, :]))

    return np.linalg.cholesky(corr_matrix + CLM_DP_Experiment.JITTER_EPS * np.eye(num_timesteps * num_axes))

def inject_noise_into_windows_per_sensor(windows, epsilon, window_size=WINDOW_SIZE, num_axes=NUM_AXES):
    sensor_L = precompute_sensor_cholesky_decomp(window_size, num_axes)

    noised_windows = []

    for window in tqdm(windows, desc="Injecting noise", unit="window"):
        noised_window = window.copy()
        for (sensor_start_idx, sensor_end_idx) in SENSOR_RANGE_IDX.values():
            sensor_data = window[:, sensor_start_idx:sensor_end_idx]
            flat_sensor_data = sensor_data.reshape(-1) # shape = (NUM_TIMESTEPS x NUM_AXES,)
            noised_flat_sensor_data = flat_sensor_data + sample_correlated_laplace_noise(sensor_L, epsilon)
            noised_sensor_data = noised_flat_sensor_data.reshape(window_size, num_axes)

            noised_window[:, sensor_start_idx:sensor_end_idx] = noised_sensor_data

        noised_windows.append(noised_window)
    
    return np.stack(noised_windows)            

def append_to_log(log_path, epsilon, decay_factor, jitter_eps, X_shape, output_dir):
    """
    Appends metadata about the CLM-DP noise injection to a CSV log file.

    Args:
        log_path (str or Path): Path to the CSV log file.
        epsilon (float): Privacy budget used.
        decay_factor (float): Correlation decay factor for the covariance matrix.
        jitter_eps (float): Jitter added to ensure positive-definiteness.
        X_shape (tuple): Shape of the IMU input array (num_windows, window_len, num_features).
        output_dir (Path): Directory where the noised output was saved.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp_now = get_timestamp_now()
    log_exists = os.path.isfile(log_path)

    with open(log_path, 'a', newline="") as csv_file:
        writer = csv.writer(csv_file)
        if not log_exists:
            writer.writerow([
                "epsilon", "decay_factor", 
                "jitter_eps", "num_windows", 
                "window_shape", "flattened_dim", 
                "output_dir", "timestamp"
            ])
        
        writer.writerow([
            epsilon, 
            decay_factor, 
            jitter_eps, 
            X_shape[0],                         # num_windows
            f"({X_shape[1]}, {X_shape[2]})",    # window_shape
            X_shape[1] * X_shape[2],            # flattened_dim
            str(output_dir), 
            timestamp_now
        ])

def main():
    """
    Loads clean IMU data windows and injects temporally correlated Laplace noise into each window
    using the Correlated Laplace Mechanism (CLM) to achieve local differential privacy.

    Steps:
        1. Load preprocessed IMU windows and labels from .npy files.
        2. Inject CLM-DP noise into each sensor segment of each window.
        3. Save the noised data and corresponding labels to a directory named for the ε value.

    Output:
        Saves noised windows and unchanged labels in a subdirectory of `data/windows`:
            - data/windows/clm_dp_eps_{ε}/X_windows.npy
            - data/windows/clm_dp_eps_{ε}/y_labels.npy
        Saves metadata values for noise injection at each epsilon value to log:
            - logs/noise_injection_log.csv
    """

    print_with_timestamp("Loading Cleaned IMU Windows and Labels...")
    X = np.load(X_PATH)
    y_fall = np.load(Y_FALL_PATH)
    Y_identity = np.load(Y_IDENTITY_PATH)

    print_with_timestamp(f"Loaded {X.shape[0]} windows of shape {X.shape[1:]}.")

    for epsilon in sorted(CLM_DP_Experiment.EPSILON_VALS):
        clm_dp_output_dir_path = WINDOWS_DIR_PATH / f"clm_dp_eps_{epsilon}"
        windows_file_path = clm_dp_output_dir_path / WINDOWS_FILE_NAME
        fall_labels_file_path = clm_dp_output_dir_path / FALL_LABELS_FILE_NAME
        identity_labels_file_path = clm_dp_output_dir_path / IDENTITY_LABELS_FILE_NAME

        if SKIP_EXISTING and all(path.exists() for path in [clm_dp_output_dir_path, windows_file_path, fall_labels_file_path, identity_labels_file_path]):
            print_color_text_with_timestamp(f"⚠️Skipping {bold_text(f'ε = {epsilon}')}: files already exist at {clm_dp_output_dir_path}", "YELLOW")
            continue
        
        # Alternative: To test uniform CLM-DP noise injection across entire window
        # L = precompute_cholesky_decomp_uniform(X.shape[1] * X.shape[2])
        # X_noised = inject_noise_into_windows(X, epsilon, L)

        X_noised = inject_noise_into_windows_per_sensor(X, epsilon, WINDOW_SIZE, NUM_AXES)

        os.makedirs(clm_dp_output_dir_path, exist_ok=True)
        X_noised_reshaped = X_noised.reshape(X.shape)  # Reshape back to (num_windows, 200, 9)
        np.save(windows_file_path, X_noised_reshaped)
        np.save(fall_labels_file_path, y_fall)
        np.save(identity_labels_file_path, Y_identity)    

        print_color_text_with_timestamp(f"✅Saved {X_noised_reshaped.shape[0]} noised windows at ε = {epsilon} to:", "RED")
        print_color_text(clm_dp_output_dir_path, "BLUE")

        append_to_log(LOG_FILE_PATH, epsilon, CLM_DP_Experiment.DECAY_FACTOR, CLM_DP_Experiment.JITTER_EPS, X.shape, clm_dp_output_dir_path)

    print()
    print_color_text_with_timestamp(f"Noise Injection Log saved to:", "RED")
    print_color_text(LOG_FILE_PATH, "BLUE")

if __name__ == '__main__':
    set_seed() # Default seed value is 42
    main()