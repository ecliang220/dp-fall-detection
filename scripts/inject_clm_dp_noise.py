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
import sys
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add project root to sys.path so `util` functions can be found
sys.path.append(str(Path(__file__).resolve().parents[1]))
from util.util import print_with_timestamp

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
# Experiment: CLM Differential Privacy Config
# --------------------------------------------------------------------
EPSILON = 1.0
DECAY_FACTOR = 0.95
# Numerical jitter added to the diagonal to ensure positive-definiteness
JITTER_EPS = 1e-6

# --------------------------------------------------------------------
# Directory and File Paths
# --------------------------------------------------------------------
# Directory Path for Training Windows
WINDOWS_DIR_PATH = PROJECT_ROOT / "data/windows"
# Directory Path for Training Windows with CLM Differential Privacy
CLM_DP_OUTPUT_DIR_PATH = WINDOWS_DIR_PATH / f"clm_dp_eps_{EPSILON}"
# Training model input file names
WINDOWS_FILE_NAME = "X_windows.npy"
LABELS_FILE_NAME = "y_labels.npy"
# Training model input files: windowed IMU data and binary labels in .npy format
X_PATH = WINDOWS_DIR_PATH / WINDOWS_FILE_NAME
Y_PATH = WINDOWS_DIR_PATH / LABELS_FILE_NAME

def flatten_windows_to_1d(X_windows):
    """
    Flattens each 2D sensor window into a 1D vector.

    Args:
        X_windows (np.ndarray): A 3D NumPy array of shape (num_windows, time_steps, num_features),
                                typically representing fixed-length IMU data windows.

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

def inject_noise_into_windows(windows):
    X_flat = flatten_windows_to_1d(windows)  # shape: (num_windows, 1800)

    X_noised = np.zeros_like(X_flat)

    # Precompute Cholesky decomposition of exponential decay covariance matrix
    indices = np.arange(X_flat.shape[1])
    corr_matrix = np.power(DECAY_FACTOR, np.abs(indices[:, None] - indices[None, :]))
    L = np.linalg.cholesky(corr_matrix + JITTER_EPS * np.eye(X_flat.shape[1]))

    print_with_timestamp(f"Injecting CLM DP noise at ε = {EPSILON} into windows...")
    
    for i in tqdm(range(X_flat.shape[0]), desc="Injecting noise", unit="window"): # Use progress bar
        # Inject CLM noise into each window
        noise = sample_correlated_laplace_noise(L, epsilon=EPSILON)
        X_noised[i] = X_flat[i] + noise
    
    return X_noised

def main():
    """
    Loads clean IMU data windows and injects temporally correlated Laplace noise into each window
    using the Correlated Laplace Mechanism (CLM) to achieve local differential privacy.

    Steps:
        1. Load preprocessed IMU windows and labels from .npy files.
        2. Flatten each window to a 1D vector.
        3. Inject CLM noise into each vectorized window using the Gaussian trick.
        4. Reshape noised data back to 3D (num_windows, time_steps, num_features).
        5. Save the noised data and corresponding labels to a directory named for the ε value.

    Output:
        Saves noised windows and unchanged labels in a subdirectory of `data/windows`:
            - data/windows/clm_dp_eps_{ε}/X_windows.npy
            - data/windows/clm_dp_eps_{ε}/y_labels.npy
    """

    print_with_timestamp("Loading Cleaned IMU Windows and Labels...")
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    print_with_timestamp(f"Loaded {X.shape[0]} windows of shape {X.shape[1:]}...")
    
    os.makedirs(CLM_DP_OUTPUT_DIR_PATH, exist_ok=True)

    X_noised = inject_noise_into_windows(X)

    X_noised_reshaped = X_noised.reshape(X.shape)  # Reshape back to (num_windows, 200, 9)
    np.save(CLM_DP_OUTPUT_DIR_PATH / WINDOWS_FILE_NAME, X_noised_reshaped)
    np.save(CLM_DP_OUTPUT_DIR_PATH / LABELS_FILE_NAME, y)    

    print_with_timestamp(f"✅ Saved {X_noised_reshaped.shape[0]} noised windows to: {CLM_DP_OUTPUT_DIR_PATH}")


if __name__ == '__main__':
    main()