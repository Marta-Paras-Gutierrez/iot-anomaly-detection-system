"""
=========================================================
IoT Traffic Monitoring & Anomaly Detection System
=========================================================

Module: 03 - Feature Extraction

Description:

This module extracts statistical traffic features from
cleaned packet DataFrames.

Packets are grouped into fixed time windows in order
to generate Machine Learning datasets.

Workflow:

1. Load cleaned DataFrame
2. Create time windows
3. Compute statistical features
4. Export feature dataset

Input: data/cleaned/

Output: data/datasets/

Author: Marta Parás
"""

# =========================================================
# Imports
# =========================================================

# File management
from pathlib import Path

# Numerical operations
import numpy as np

# Data manipulation
import pandas as pd

# Project configuration
from src.config import (
    PROJECT_NAME,
    CLEANED_DATA_DIR,
    DATASETS_DIR,
    TIME_WINDOW,
    CSV_SEPARATOR,
    CSV_ENCODING
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
# Clean dataset discovery
# =========================================================

def get_cleaned_files() -> list[Path]:
    """
    Search every cleaned CSV file.

    Returns
    -------
    list[Path]
    """

    return sorted(CLEANED_DATA_DIR.glob("*.csv"))

# =========================================================
# Load DataFrame
# =========================================================

def load_dataframe(file_path: Path) -> pd.DataFrame:
    """
    Load a cleaned DataFrame.

    Parameters
    ----------
    file_path : Path

    Returns
    -------
    pd.DataFrame
    """

    logger.info(f"Loading: {file_path.name}")

    dataframe = pd.read_csv(
        file_path,
        sep=CSV_SEPARATOR,
        encoding=CSV_ENCODING
    )

    logger.info(f"Packets loaded: {len(dataframe)}")

    return dataframe

# =========================================================
# Create time windows
# =========================================================

def create_time_windows(dataframe: pd.DataFrame):
    """
    Group packets into fixed time windows.

    Parameters
    ----------
    dataframe : pd.DataFrame

    Returns
    -------
    GroupBy
    """

    dataframe = dataframe.copy()

    initial_time = dataframe["timestamp"].min()
    
    dataframe["time_window"] = ((dataframe["timestamp"] - initial_time) // TIME_WINDOW).astype(int)

    return dataframe.groupby("time_window")

# =========================================================
# Packet statistics
# =========================================================

def calculate_packet_statistics(window: pd.DataFrame) -> dict:
    """
    Calculate packet-related statistics.
    """

    return {
        "total_packets": len(window),
        "mean_packet_size": window["packet_length"].mean(),
        "std_packet_size": window["packet_length"].std()
    }

# =========================================================
# Protocol statistics
# =========================================================

def calculate_protocol_statistics(window: pd.DataFrame) -> dict:
    """
    Calculate protocol counters.
    """

    return {
        "icmp_count": (window["protocol"] == "ICMP").sum(),
        "arp_count": (window["protocol"] == "ARP").sum()
    }

# =========================================================
# Inter-arrival statistics
# =========================================================

def calculate_interarrival_statistics(window: pd.DataFrame) -> dict:
    """
    Calculate packet inter-arrival metrics.
    """

    intervals = (window["timestamp"].diff().fillna(0))

    return {
        "mean_inter_arrival": intervals.mean(),
        "std_inter_arrival": intervals.std()
    }

# =========================================================
# Feature extraction
# =========================================================

def extract_features(grouped_packets) -> pd.DataFrame:
    """
    Extract statistical features from every
    time window.

    Parameters
    ----------
    grouped_packets : DataFrameGroupBy

    Returns
    -------
    pd.DataFrame
    """

    feature_rows = []

    for window_id, window in grouped_packets:
        features = {"time_window": window_id}

        features.update(calculate_packet_statistics(window))
        features.update(calculate_protocol_statistics(window))
        features.update(calculate_interarrival_statistics(window))

        feature_rows.append(features)

    features_dataframe = pd.DataFrame(feature_rows)

    logger.info(f"Time windows generated: {len(features_dataframe)}")

    return features_dataframe

# =========================================================
# Process a single DataFrame
# =========================================================

def process_dataframe(file_path: Path) -> int:
    """
    Complete feature extraction workflow.

    Parameters
    ----------
    file_path : Path

    Returns
    -------
    int
        Number of generated time windows.
    """

    print_separator(logger, "-")

    logger.info(f"Processing: {file_path.name}")

    dataframe = load_dataframe(file_path)

    grouped_packets = create_time_windows(dataframe)

    feature_dataframe = extract_features(grouped_packets)

    base_name = get_base_filename(file_path)

    output_file = (DATASETS_DIR / f"{base_name}_dataset.csv")

    overwrite_file(
        output_file,
        logger
    )

    save_dataframe(
        feature_dataframe,
        output_file,
        CSV_SEPARATOR,
        CSV_ENCODING
    )

    logger.info(f"Feature dataset generated: {output_file.name}")

    logger.info("Feature extraction completed.\n")

    return len(feature_dataframe)

# =========================================================
# Main workflow
# =========================================================

def main():
    """
    Execute the feature extraction pipeline.
    """

    start_time = start_timer()

    print_separator(logger)
    logger.info(PROJECT_NAME)
    print_separator(logger)

    cleaned_files = get_cleaned_files()

    if not cleaned_files:
        logger.warning("No cleaned datasets were found.")
        return

    logger.info(f"Datasets detected: {len(cleaned_files)}\n")

    successful = 0
    failed = 0
    total_windows = 0

    for file_path in cleaned_files:
        try:
            windows = process_dataframe(file_path)
            total_windows += windows
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
            "Packets": total_windows,
            "Execution": f"{execution_time:.2f} s",
            "Output": DATASETS_DIR
        }
    )

# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()