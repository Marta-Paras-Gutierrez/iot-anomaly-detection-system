"""
=========================================================
IoT Traffic Monitoring & Anomaly Detection System
=========================================================

Module: Dashboard Utilities

Description:

Utility functions used by the Streamlit dashboard.

These functions centralize the loading of datasets,
models and detection results in order to keep the
dashboard implementation simple and reusable.

Input:
    data/
    results/
    models/

Author: Marta Parás
"""

# =========================================================
# Imports
# =========================================================

# Data manipulation
import pandas as pd

# Project configuration
from src.config import (
    RESULTS_DIR,
    DETECTION_RESULTS_DIR,
    CSV_SEPARATOR,
    CSV_ENCODING
)

# Utilities
from src.utils import load_dataframe

# =========================================================
# Model comparison
# =========================================================

def load_model_results() -> pd.DataFrame:
    """
    Load model evaluation results.
    """

    return load_dataframe(
        RESULTS_DIR / "model_comparison.csv",
        separator=CSV_SEPARATOR,
        encoding=CSV_ENCODING
    )

# =========================================================
# Detection results
# =========================================================

def load_detection_results() -> list[pd.DataFrame]:
    """
    Load every detection result.
    """

    datasets: list[pd.DataFrame] = []

    for file_path in sorted(DETECTION_RESULTS_DIR.glob("*_detections.csv")):

        dataframe = load_dataframe(
            file_path,
            separator=CSV_SEPARATOR,
            encoding=CSV_ENCODING
        )

        capture_name  = file_path.stem.replace("_detections", "")
        
        dataframe["capture"] = capture_name

        datasets.append(dataframe)

    return datasets

# =========================================================
# Detection statistics
# =========================================================

def get_detection_statistics() -> dict[str, int | float]:
    """
    Calculate global detection statistics.
    """

    datasets = load_detection_results()

    total_captures = len(datasets)

    total_windows = 0

    total_anomalies = 0

    for dataframe in datasets:
        
        total_windows += len(dataframe)

        total_anomalies += dataframe["final_prediction"].sum()

    anomaly_percentage = (
        (total_anomalies / total_windows) * 100
        if total_windows > 0
        else 0
    )

    return {
        "captures": total_captures,
        "windows": total_windows,
        "anomalies": total_anomalies,
        "percentage": anomaly_percentage
    }

# =========================================================
# Best Machine Learning model
# =========================================================

def get_best_model() -> dict[str, str | float]:
    """
    Return the best evaluated model.
    """

    dataframe = load_model_results()

    best = dataframe.loc[dataframe["F1-Score"].idxmax()]

    return {
        "name": best["Model"],
        "accuracy": best["Accuracy"],
        "precision": best["Precision"],
        "recall": best["Recall"],
        "f1_score": best["F1-Score"]
    }

# =========================================================
# Available captures
# =========================================================

def get_capture_names() -> list[str]:
    """
    Return every analysed capture.
    """

    return sorted([
        file_path.stem.replace("_detections", "")

        for file_path in DETECTION_RESULTS_DIR.glob("*_detections.csv")
    ])

# =========================================================
# Load one detection result
# =========================================================

def load_capture(capture_name: str) -> pd.DataFrame:
    """
    Load one analysed capture.
    """

    file_path = DETECTION_RESULTS_DIR / (f"{capture_name}_detections.csv")

    return load_dataframe(
        file_path,
        separator=CSV_SEPARATOR,
        encoding=CSV_ENCODING
    )