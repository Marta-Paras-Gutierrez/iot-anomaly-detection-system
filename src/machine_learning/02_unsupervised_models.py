"""
=========================================================
IoT Traffic Monitoring & Anomaly Detection System
=========================================================

Module: 02 - Unsupervised Machine Learning

Description:

This module trains unsupervised Machine Learning models
for anomaly detection using the extracted traffic features.

Workflow:

1. Load feature datasets
2. Assign labels
3. Merge datasets
4. Prepare feature matrix
5. Train Isolation Forest
6. Save trained model

Input: data/datasets/

Output: models/

Author: Marta Parás
"""

# =========================================================
# Imports
# =========================================================

# File management
from pathlib import Path

# Data manipulation
import pandas as pd

# Machine Learning
from sklearn.ensemble import IsolationForest

# Project configuration
from src.config import (
    PROJECT_NAME,
    DATASETS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
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
    save_model
)

# =========================================================
# Logger configuration
# =========================================================

logger = get_logger(__name__)

# =========================================================
# Dataset discovery
# =========================================================

def get_dataset_files() -> list[Path]:
    """
    Search all feature datasets.
    """

    return sorted(DATASETS_DIR.glob("*_dataset.csv"))

# =========================================================
# Dataset loading
# =========================================================

def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load one dataset.
    """

    logger.info(f"Loading dataset: {file_path.name}")

    dataframe = pd.read_csv(
        file_path,
        sep=CSV_SEPARATOR,
        encoding=CSV_ENCODING
    )

    logger.info(f"Samples loaded: {len(dataframe)}")

    return dataframe

# =========================================================
# Dataset label assignment
# =========================================================

def assign_label(dataframe: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    """
    Assign labels only for evaluation.

    Isolation Forest ignores them
    during training.
    """

    dataframe = dataframe.copy()

    filename = get_base_filename(file_path).lower()

    if "normal" in filename:
        dataframe["label"] = 0
    else:
        dataframe["label"] = 1

    return dataframe

# =========================================================
# Merge datasets
# =========================================================

def merge_datasets(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge every dataset.
    """

    dataframe = pd.concat(
        datasets,
        ignore_index=True
    )

    logger.info(f"Total samples: {len(dataframe)}")

    return dataframe

# =========================================================
# Feature preparation
# =========================================================

def prepare_features(dataframe: pd.DataFrame):
    """
    Separate features and labels.
    """

    X = dataframe.drop(columns=["label"])

    y = dataframe["label"]

    return X, y

# =========================================================
# Isolation Forest training
# =========================================================

def train_isolation_forest(X: pd.DataFrame) -> IsolationForest:
    """
    Train the Isolation Forest model.
    """

    logger.info(f"Training Isolation Forest...")

    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=RANDOM_STATE
    )

    model.fit(X)

    logger.info("Model trained successfully.")

    return model

# =========================================================
# Main workflow
# =========================================================

def main():
    """
    Execute the unsupervised learning pipeline.
    """

    start_time = start_timer()

    print_separator(logger)
    logger.info(PROJECT_NAME)
    print_separator(logger)

    dataset_files = get_dataset_files()

    if not dataset_files:
        logger.warning("No feature datasets were found.")
        return

    logger.info(f"Datasets detected: {len(dataset_files)}\n")

    datasets = []

    for file_path in dataset_files:
        dataframe = load_dataset(file_path)

        dataframe = assign_label(
            dataframe,
            file_path
        )

        datasets.append(dataframe)

    merged_dataset = merge_datasets(datasets)

    X, y = prepare_features(merged_dataset)

    logger.info(f"Training samples: {len(X)}")

    model = train_isolation_forest(X)

    save_model(
        model,
        MODELS_DIR / "isolation_forest_model.joblib",
        logger
    )

    execution_time = stop_timer(start_time)

    print_summary(
        logger,
        "PROCESS SUMMARY",
        {
            "Datasets": len(dataset_files),
            "Samples": len(X),
            "Execution": f"{execution_time:.2f} s",
            "Output": MODELS_DIR
        }
    )

# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()