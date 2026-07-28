# =======================================================
# Imports
# =======================================================
# Logging
import logging

# Time measurement
import time

# File and directory management
from pathlib import Path

# Data manipulation
import pandas as pd

# Model serialization
import joblib

# =======================================================
# =================== File utilities ====================
# =======================================================

# =======================================================
# Change names
# =======================================================

# Normalize generated filenames across the processing pipeline.
def get_base_filename(file_path: Path) -> str:
    """
    Return the original capture name removing
    any pipeline suffix.

    Examples
    --------
    normal_domestico_processed.csv
            ↓
    normal_domestico_cleaned.csv

    normal_domestico_cleaned.csv
            ↓
    normal_domestico_dataset.csv
    """

    filename = file_path.stem

    suffixes = (
        "_processed",
        "_cleaned",
        "_dataset"
    )

    for suffix in suffixes:
        filename = filename.replace(suffix, "")

    return filename

# =======================================================
# Overwrite existing files
# =======================================================

def overwrite_file(file_path: Path, logger: logging.Logger) -> None:
    """
    Remove an existing file before saving.
    """

    # Remove previous file if it already exists.
    if file_path.exists():

        logger.info(f"Existing file detected: {file_path.name}")

        logger.info("Previous file will be overwritten.")

        file_path.unlink()

# =======================================================
# Capture discovery
# =======================================================

def get_capture_files(captures_dir: Path, supported_formats: list[str]) -> list[Path]:
    """
    Search all supported packet capture files.
    """

    capture_files: list[Path] = []

    for extension in supported_formats:
        capture_files.extend(
            captures_dir.glob(f"*{extension}")
        )

    return sorted(capture_files)

# =======================================================
# ======================= Logging =======================
# =======================================================

# =======================================================
# Logger configuration
# =======================================================

def get_logger(name: str) -> logging.Logger:
    """
    Create a standard project logger.
    """

    logger = logging.getLogger(name)

    if not logger.handlers:

        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s | %(message)s"
        )

    return logger

# =======================================================
# Console helpers
# =======================================================

def print_separator(logger: logging.Logger, character: str = "=", length: int = 60) -> None:
    """
    Print a formatted separator.
    """

    logger.info(character * length)

# =======================================================
# Process summary
# =======================================================

def print_summary(logger: logging.Logger, title: str, values: dict[str, object]) -> None:
    """
    Print the execution summary.
    """

    print_separator(logger)
    logger.info(title)
    print_separator(logger)

    for key, value in values.items():
        logger.info(f"{key:<12}: {value}")

    print_separator(logger)

# =======================================================
# ======================= Timing ========================
# =======================================================

# =======================================================
# Execution timer
# =======================================================

def start_timer() -> float:
    """
    Start execution timer.

    Returns
    -------
    float
        Initial timestamp.
    """

    return time.perf_counter()

# =======================================================
# Stop timer
# =======================================================

def stop_timer(start_time: float) -> float:
    """
    Stop execution timer.

    Parameters
    ----------
    start_time : float
        Initial timestamp.

    Returns
    -------
    float
        Execution time in seconds.
    """

    return time.perf_counter() - start_time

# =======================================================
# ====================== DataFrame ======================
# =======================================================

# =======================================================
# Load DataFrame
# =======================================================

def load_dataframe(file_path: Path, separator: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    """
    Load a CSV file.
    """

    dataframe = pd.read_csv(
        file_path,
        sep=separator,
        encoding=encoding
    )

    return dataframe

# =======================================================
# Save DataFrame
# =======================================================

def save_csv(dataframe: pd.DataFrame, output_file: Path, logger, separator: str = ",", encoding: str = "utf-8") -> Path:
    """
    Save a DataFrame as CSV.
    """

    overwrite_file(
        output_file,
        logger
    )

    dataframe.to_csv(
        output_file,
        index=False,
        sep=separator,
        encoding=encoding
    )

    logger.info(f"DataFrame saved: {output_file.name}")

    return output_file

# =======================================================
# ======================= Models ========================
# =======================================================

# =======================================================
# Load Machine Learning model
# =======================================================

def load_model(model_path: Path, logger: logging.Logger):
    """
    Load a trained Machine Learning model.
    """

    model = joblib.load(model_path)

    logger.info(f"Model loaded: {model_path.name}")

    return model

# =======================================================
# Save trained Machine Learning model
# =======================================================

def save_model(model, output_file: Path, logger) -> Path:
    """
    Save a trained Machine Learning model.

    Parameters
    ----------
    model
        Trained model instance.

    output_file : Path
        Destination file.

    logger : logging.Logger
        Project logger.

    Returns
    -------
    Path
        Saved model path.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    overwrite_file(
        output_file,
        logger
    )
    
    joblib.dump(
        model,
        output_file
    )

    logger.info(f"Model saved: {output_file.name}")

    return output_file

# =======================================================
# ====================== Validation =====================
# =======================================================

