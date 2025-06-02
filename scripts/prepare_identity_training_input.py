import pandas as pd
from pandas.errors import ParserError
from pathlib import Path
import numpy as np
import os

# Directory containing preprocessed CSV files (1 file = 1 trial)
ROOT_INPUT_DIR_PATH = Path("../data/preprocessed")

# Directory to output saved window arrays and labels
OUTPUT_DIRECTORY_PATH = Path("../data/windows")
OUTPUT_FILE_PATH_WINDOWS = OUTPUT_DIRECTORY_PATH / "X_identity_windows.npy"
OUTPUT_FILE_PATH_LABELS = OUTPUT_DIRECTORY_PATH / "y_identity_labels.npy"

# Fixed-length sliding window parameters
WINDOW_SIZE = 200           # 200 rows = 1 second at 200 Hz
WINDOW_INTERVAL = 100       # 50% overlap (stride = 100)

# Number of Adult participants
NUM_ADULTS = 23

# Number of Elder participants
NUM_ELDERS = 15

def prepare_windows_for_file(file_path):
    """
    Reads a preprocessed CSV file and yields overlapping windows of fixed size.

    For each file:
    - Reads in chunks of 100 rows (to match stride)
    - Maintains a rolling buffer of rows
    - Yields 200-row overlapping windows (50% overlap)

    Args:
        file_path (Path): Path to the input CSV file

    Yields:
        np.ndarray: A 2D NumPy array of shape (200, 9) containing sensor values for one window.
    """
    buffer = []
    try:
        for chunk in pd.read_csv(file_path, chunksize=WINDOW_INTERVAL, usecols=None):
            rows = chunk.to_numpy()
            buffer.extend(rows)

            while len(buffer) >= WINDOW_SIZE:
                yield np.array(buffer[:WINDOW_SIZE])
                buffer = buffer[WINDOW_INTERVAL:]

    except ParserError as e:
        print(f"ParserError in file {file_path}: {e}")
        return None
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError in file {file_path}: {e}")
        return None
    except Exception as e:
        print(f"General error in file {file_path}: {e}")
        return None

def prepare_all_files(root_directory_path):
    """
    Processes all CSV trial files in a directory into labeled sliding windows.

    For each `.csv` file:
    - Infers the participant ID from the second field in the filename (e.g., 'SA02', 'SE15')
    - Extracts overlapping 200-row windows with 50% overlap
    - Stores all windows and labels (participant ID) in two separate arrays

    The final arrays are saved as `.npy` files for fast loading later.

    Args:
        root_directory_path (Path): Root directory containing CSV subfolders.

    Returns:
        None
    """
    OUTPUT_DIRECTORY_PATH.mkdir(parents=True, exist_ok=True)

    X = []
    y = []

    for dir_path, _, file_names in os.walk(root_directory_path):
        if Path(dir_path) == root_directory_path:
            continue
        for file_name in sorted(file_names):
            if file_name.endswith('.csv'):
                file_path = Path(dir_path) / file_name

                participant_tag = file_name.split("_")[1]   # e.g. 'SA02' or 'SE15'
                participant_type = participant_tag[:2]
                participant_num = int(participant_tag[2:])  # e.g. '02' or '15'

                if participant_type == "SA":
                    participant_id = participant_num                     # SA01–SA23 → 1–23
                elif participant_type == "SE":
                    participant_id = NUM_ADULTS + participant_num        # SE01–SE15 → 24–38
                else:
                    print(f"Unknown participant tag in {file_name}")
                    continue
                label = participant_id - 1  # Starts at 0

                new_windows_gen = prepare_windows_for_file(file_path)
                if new_windows_gen is not None:
                    new_windows = list(new_windows_gen)
                    if new_windows:
                        X.extend(new_windows)
                        y.extend([label] * len(new_windows))
                        print(f"Processed {file_name}: {len(new_windows)} windows")

    # Convert to NumPy arrays and save
    X = np.array(X)
    y = np.array(y, dtype=np.int8)
    np.save(OUTPUT_FILE_PATH_WINDOWS, X)
    np.save(OUTPUT_FILE_PATH_LABELS, y)

    print("_______________________________________________________")
    print(f"Saved {X.shape[0]} windows to {OUTPUT_FILE_PATH_WINDOWS}")
    print(f"Saved {y.shape[0]} labels to {OUTPUT_FILE_PATH_LABELS}")

    # Participant label distribution summary
    unique, counts = np.unique(y, return_counts=True)
    print("\nParticipant distribution (label → window count):")
    for uid, count in zip(unique, counts):
        print(f"  {uid:2d}: {count}")

def main():
    """
    Main entry point: Slices SisFall CSV files into 1s windows with 50% overlap
    and outputs training-ready NumPy arrays for CNN-based identity inference.
    """
    prepare_all_files(ROOT_INPUT_DIR_PATH)

if __name__ == '__main__':
    main()
