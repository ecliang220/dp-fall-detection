import csv
import os
from pathlib import Path

# Raw SisFall data input path
ROOT_INPUT_DIR_PATH = Path("../data/raw")
# Preprocessed SisFall data output path
ROOT_OUTPUT_DIR_PATH = Path("../data/preprocessed")

# Number of sensors used in SisFall dataset
NUM_SENSORS = 3

# Interval between each sensor measurement (proportional to 200 Hz)
MEASUREMENT_INTERVAL = 1 / 200  # 0.005 seconds per sample

# Sensor characteristics
ADXL345_RES = 13
ADXL345_RANGE = 16
ITG3200_RES = 16
ITG3200_RANGE = 2000
MMA8451Q_RES = 14
MMA8451Q_RANGE = 8

sensors_order = {
    "ADXL345": 0,
    "ITG3200": 1,
    "MMA8451Q": 2
}

axes_order = {
    'x': 0,
    'y': 1,
    'z': 2
}

class Sensor:
    """
    Represents a motion sensor used in the SisFall dataset, including its identifier, 
    resolution in bits, and physical measurement range.

    Attributes:
        id (str): Sensor name (e.g., "ADXL345").
        resolution_value (int): ADC resolution in bits (e.g., 13).
        range_value (float): Full-scale range for measurement (e.g., ±16 g).
    """
    def __init__(self, id, resolution_value, range_value):
        self.id = id
        self.resolution_value = resolution_value
        self.range_value = range_value

motion_sensors = {
    "ADXL345": Sensor("ADXL345", ADXL345_RES, ADXL345_RANGE),
    "ITG3200": Sensor("ITG3200", ITG3200_RES, ITG3200_RANGE),
    "MMA8451Q": Sensor("MMA8451Q", MMA8451Q_RES, MMA8451Q_RANGE)
}

def convert_raw_to_physical(bits, sensor_range, sensor_resolution):
    """
    Converts raw sensor output (bits) to physical units using the sensor's range and resolution.

    Args:
        bits (int): Raw digital value from the sensor.
        sensor_range (float): Full-scale measurement range (e.g. 16 for ±16g).
        sensor_resolution (int): Sensor resolution in bits (e.g. 13, 14, or 16).

    Returns:
        float: Value in physical units (e.g. g or °/s).
    """
    return bits * (2 * sensor_range / 2 ** sensor_resolution)

def preprocess_single_file(infile_path, outdir_path):
    """
    Reads a single SisFall trial file, converts raw data sensor measurements from 
    bits to physical units, and writes the preprocessed data to a new CSV file.

    For each line in the original `.txt` file:
    - Parses and writes the 9 sensor measurements
    - Handles and reports malformed lines if any errors occur during conversion

    Args:
        infile_path (str): Path to the original `.txt` file containing raw sensor data
        outdir_path (str): Directory where the new preprocessed `.csv` file will be written

    Returns:
        None
    """
    file_name = Path(infile_path).stem
    outdir_path.mkdir(parents=True, exist_ok=True)
    outfile_path = outdir_path / f"{file_name}.csv"

    with open(infile_path, "r") as infile, open(outfile_path, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["ADXL345_x", "ADXL345_y", "ADXL345_z",
                         "ITG3200_x", "ITG3200_y", "ITG3200_z",
                         "MMA8451Q_x", "MMA8451Q_y", "MMA8451Q_z"])

        for i, line in enumerate(infile):
            if not line.strip():
                continue
            try:
                measurements = line.strip().strip(";").split(",")
                for sensor_id, sensor_num in sensors_order.items():
                    col_idx_start = sensor_num * NUM_SENSORS
                    for _, addIdx in axes_order.items():
                        raw = float(measurements[col_idx_start + addIdx])
                        physical = convert_raw_to_physical(
                            raw,
                            motion_sensors[sensor_id].range_value,
                            motion_sensors[sensor_id].resolution_value
                        )
                        measurements[col_idx_start + addIdx] = physical
                writer.writerow(measurements)
            except Exception as e:
                print(f"Error at line {i + 1} in file {infile_path}: {e}")
                print(f"Line content: {line.strip()}")

def preprocess_files(root_in_dir_path, root_out_dir_path):
    """
    ONLY NEEDS TO BE RUN ONCE OR IF INPUT DATA CHANGES!!!
    Preprocesses all trial files from subdirectories in the raw SisFall dataset 
    and writes `.csv` files to the corresponding output directory. 

    For each subdirectory (representing a subject):
    - Creates a mirrored directory in the output folder
    - Converts all `.txt` trial files 

    Args:
        root_in_dir_path (str): Path to the root directory containing original trial `.txt` files
        root_out_dir_path (str): Path to the root directory where processed `.csv` files will be saved

    Returns:
        None
    """
    for dir_path, _, file_names in os.walk(root_in_dir_path):
        current_dir = Path(dir_path)
        if current_dir == root_in_dir_path:
            continue
        relative_dir = current_dir.relative_to(root_in_dir_path)
        output_dir = root_out_dir_path / relative_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        for file_name in file_names:
            if file_name.endswith(".txt"):
                preprocess_single_file(current_dir / file_name, output_dir)

def main():
    """
    Main entry point for preprocessing all SisFall raw `.txt` trial files.

    This function:
    - Traverses the raw dataset directory
    - Converts each file into `.csv` with physical units
    - Writes the converted files into the corresponding output directory

    Only needs to be run once unless the input data changes.
    """
    preprocess_files(ROOT_INPUT_DIR_PATH, ROOT_OUTPUT_DIR_PATH)

if __name__ == "__main__":
    main()
