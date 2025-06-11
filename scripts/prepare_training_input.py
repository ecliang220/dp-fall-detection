import pandas as pd
from pandas.errors import ParserError
from pathlib import Path
import numpy as np
import os
import sys

# Add project root to sys.path so `util` functions can be found
sys.path.append(str(Path(__file__).resolve().parents[1]))
from util.util import print_color_text_with_timestamp, print_color_text
from model.model_util import WINDOWS_DIR_PATH, X_PATH, Y_FALL_PATH, Y_IDENTITY_PATH, PREPROCESSED_DIR_PATH
from model.model_util import WINDOW_SIZE, WINDOW_INTERVAL, NUM_ADULTS

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
    WINDOWS_DIR_PATH.mkdir(parents=True, exist_ok=True)

    X = []
    y_fall = []
    y_identity = []

    print_color_text_with_timestamp("Preparing windows for Fall and Identity Training Input...", "RED")
    for dir_path, _, file_names in os.walk(root_directory_path):
        if Path(dir_path) == root_directory_path:
            continue
        for file_name in sorted(file_names):
            if file_name.endswith('.csv'):
                file_path = Path(dir_path) / file_name

                # Determine label from file name prefix
                activity_code = file_name.split("_")[0]
                fall_label = 1 if activity_code.startswith('F') else 0

                participant_tag = file_name.split("_")[1]   # e.g. 'SA02' or 'SE15'
                participant_type = participant_tag[:2]
                participant_num = int(participant_tag[2:])  # e.g. '02' or '15'

                if participant_type == "SA":
                    participant_id = participant_num                     # SA01–SA23 → 1–23
                elif participant_type == "SE":
                    participant_id = NUM_ADULTS + participant_num        # SE01–SE15 → 24–38
                else:
                    print_color_text(f"Unknown participant tag in {file_name}", "YELLOW")
                    continue
                identity_label = participant_id - 1  # Starts at 0

                new_windows_gen = prepare_windows_for_file(file_path)
                if new_windows_gen is not None:
                    new_windows = list(new_windows_gen)
                    if new_windows:
                        X.extend(new_windows)
                        y_fall.extend([fall_label] * len(new_windows))
                        y_identity.extend([identity_label] * len(new_windows))
                        print_color_text_with_timestamp(f"Processed {file_name}: {len(new_windows)} windows", "BRIGHT_BLUE")

    # Convert to NumPy arrays and save
    X = np.array(X)
    y_fall = np.array(y_fall, dtype=np.int8)
    y_identity = np.array(y_identity, dtype=np.int8)
    np.save(X_PATH, X)
    np.save(Y_FALL_PATH, y_fall)
    np.save(Y_IDENTITY_PATH, y_identity)

    assert X.shape[0] == y_fall.shape[0] == y_identity.shape[0], "Mismatch in window and label counts"

    print("_______________________________________________________")
    print_color_text_with_timestamp(f"Total windows: {X.shape[0]}, Fall labels: {y_fall.shape[0]}, Identity labels: {y_identity.shape[0]}", "RED")
    print_color_text_with_timestamp(f"Saved {X.shape[0]} windows to:", "RED")
    print_color_text(str(X_PATH), "BLUE")
    print_color_text_with_timestamp(f"Saved {y_fall.shape[0]} fall labels to:", "RED")
    print_color_text(str(Y_FALL_PATH), "BLUE")
    print_color_text_with_timestamp(f"Saved {y_identity.shape[0]} identity labels to:", "RED")
    print_color_text(str(Y_IDENTITY_PATH), "BLUE")

    # Participant label distribution summary
    unique, counts = np.unique(y_identity, return_counts=True)
    print("\nParticipant distribution (label → window count):")
    for uid, count in zip(unique, counts):
        print(f"  {uid:2d}: {count}")

def main():
    """
    Main entry point: Slices SisFall CSV files into 1s windows with 50% overlap
    and outputs training-ready NumPy arrays for both:
    - Fall detection (binary labels)
    - Identity inference (participant ID labels)
    """
    prepare_all_files(PREPROCESSED_DIR_PATH)

if __name__ == '__main__':
    main()
