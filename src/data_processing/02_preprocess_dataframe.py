"""
=========================================================
IoT Traffic Monitoring & Anomaly Detection System
=========================================================

Module: 02 - Preprocess DataFrame

Description:

This module performs the initial preprocessing of the
packet DataFrames generated during the previous stage.

The objective is to clean and validate the extracted
network traffic before feature extraction.

Workflow:

1. Search processed CSV files
2. Load DataFrame
3. Validate required columns
4. Remove invalid rows
5. Remove duplicated packets
6. Sort packets chronologically
7. Export cleaned DataFrame

Input: captures/processed/

Output: data/cleaned/

Author: Marta Parás
"""

# =========================================================
# Imports
# =========================================================

# File and directory management
from pathlib import Path

# Data manipulation
import pandas as pd

# Project configuration
from src.config import (
    PROJECT_NAME,
    PROCESSED_DATA_DIR,
    CLEANED_DATA_DIR,
    REQUIRED_COLUMNS,
    CSV_SEPARATOR,
    CSV_ENCODING,
)

# Utilities
from src.utils import (
    get_base_filename,
    get_logger,
    start_timer,
    stop_timer,
    print_separator,
    print_summary,
    overwrite_file,
    save_dataframe
)

# =========================================================
# Logger configuration
# =========================================================

logger = get_logger(__name__)

# =========================================================
# Processed CSV discovery
# =========================================================

def get_processed_files() -> list[Path]:
    """
    Search every processed CSV generated during
    the previous pipeline stage.

    Returns
    -------
    list[Path]
        Sorted list containing every processed CSV.
    """

    processed_files = sorted(PROCESSED_DATA_DIR.glob("*.csv"))

    return processed_files

# =========================================================
# DataFrame loading
# =========================================================

def load_dataframe(file_path: Path) -> pd.DataFrame:
    """
    Load a processed CSV file.

    Parameters
    ----------
    file_path : Path

    Returns
    -------
    pandas.DataFrame
    """

    logger.info(f"Loading: {file_path.name}")

    dataframe = pd.read_csv(
        file_path,
        sep=CSV_SEPARATOR,
        encoding=CSV_ENCODING
    )

    logger.info(f"Rows loaded: {len(dataframe)}")

    return dataframe

# =========================================================
# Data validation
# =========================================================

def validate_dataframe(dataframe: pd.DataFrame) -> None:
    """
    Validate whether the DataFrame contains
    every required column.

    Parameters
    ----------
    dataframe : pd.DataFrame

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    logger.info("Column validation completed.")

# =========================================================
# Data cleaning
# =========================================================

def clean_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the packet DataFrame.

    Cleaning operations:
        - Remove empty rows
        - Remove duplicated packets
        - Sort by timestamp
        - Reset index

    Parameters
    ----------
    dataframe : pd.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    initial_rows = len(dataframe)

    # Remove empty rows
    dataframe = dataframe.dropna(how="all")

    # Remove duplicated packets
    dataframe = dataframe.drop_duplicates()

    # Convert timestamps
    dataframe["timestamp"] = pd.to_numeric(
        dataframe["timestamp"],
        errors="coerce"
    )

    # Remove invalid timestamps
    dataframe = dataframe.dropna(subset=["timestamp"])

    # Sort packets chronologically
    dataframe = dataframe.sort_values(by="timestamp")

    # Reset index
    dataframe = dataframe.reset_index(drop=True)

    removed_rows = initial_rows - len(dataframe)

    logger.info(f"Removed rows: {removed_rows}")
    logger.info(f"Remaining rows: {len(dataframe)}")

    return dataframe

# =========================================================
# Process a single DataFrame
# =========================================================

def process_dataframe(file_path: Path) -> int:
    """
    Execute the preprocessing workflow
    for one processed CSV.

    Parameters
    ----------
    file_path : Path

    Returns
    -------
    int
        Number of rows contained in the
        cleaned DataFrame.
    """

    print_separator(logger, "-")

    logger.info(f"Processing: {file_path.name}")

    dataframe = load_dataframe(file_path)

    validate_dataframe(dataframe)

    dataframe = clean_dataframe(dataframe)

    base_name = get_base_filename(file_path)

    output_file = (CLEANED_DATA_DIR / f"{base_name}_cleaned.csv")

    overwrite_file(
        output_file,
        logger
    )

    save_dataframe(
        dataframe,
        output_file,
        CSV_SEPARATOR,
        CSV_ENCODING
    )

    logger.info(f"Clean DataFrame saved: {output_file.name}")

    logger.info(f"DataFrame processed successfully.\n")

    return len(dataframe)

# =========================================================
# Main workflow
# =========================================================

def main():
    """
    Execute the preprocessing pipeline.
    """

    start_time = start_timer()

    print_separator(logger)
    logger.info(PROJECT_NAME)
    print_separator(logger)

    processed_files = get_processed_files()

    if not processed_files:
        logger.warning("No processed CSV files were found.")
        return

    logger.info(f"CSV files detected: {len(processed_files)}\n")

    successful = 0
    failed = 0
    total_rows = 0

    for file_path in processed_files:
        try:
            rows = process_dataframe(file_path)
            total_rows += rows
            successful += 1
        except Exception as error:
            logger.error(f"Error processing {file_path.name}")
            logger.error(error)
            failed += 1

    execution_time = stop_timer(start_time)

    print_summary(
        logger,
        "PROCESS SUMMARY",
        {
            "Processed": successful,
            "Failed": failed,
            "Packets": total_rows,
            "Execution": f"{execution_time:.2f} s",
            "Output": CLEANED_DATA_DIR
        }
    )

# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()