"""
=========================================================
IoT Traffic Monitoring & Anomaly Detection System
=========================================================

Module: 01 - Supervised Machine Learning

Description:

This module trains supervised Machine Learning models
using the feature datasets extracted from network traffic.

Workflow:

1. Load feature datasets
2. Assign class labels
3. Merge datasets
4. Split training and testing data
5. Train Random Forest classifier
6. Evaluate model
7. Save trained model

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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project configuration
from src.config import (
    PROJECT_NAME,
    DATASETS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    N_ESTIMATORS,
    MAX_DEPTH,
    CSV_SEPARATOR,
    CSV_ENCODING
)

# Utilities
from src.utils import (
    get_base_filename,
    get_logger,
    load_dataframe,
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
    Search every feature dataset.

    Returns
    -------
    list[Path]
    """

    return sorted(DATASETS_DIR.glob("*_dataset.csv"))

# =========================================================
# Dataset loading
# =========================================================

def load_feature_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load one feature dataset.
    """

    logger.info(f"Loading: {file_path.name}")

    dataframe = load_dataframe(
        file_path,
        CSV_SEPARATOR,
        CSV_ENCODING
    )

    logger.info(f"Samples loaded: {len(dataframe)}")

    return dataframe

# =========================================================
# Label assignment
# =========================================================

def assign_label(dataframe: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    """
    Assign binary labels.

    Normal traffic:     0

    Anomalous traffic:  1
    """

    dataframe = dataframe.copy()

    filename = get_base_filename(file_path).lower()

    label = 0 if "normal" in filename else 1

    dataframe["label"] = label

    logger.info(f"Assigned label: {label}")

    return dataframe

# =========================================================
# Dataset merging
# =========================================================

def merge_datasets(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge every labelled dataset.
    """

    merged_dataframe = pd.concat(
        datasets,
        ignore_index=True
    )

    logger.info(f"Total samples: {len(merged_dataframe)}")

    return merged_dataframe

# =========================================================
# Train/Test split
# =========================================================

def split_dataset(dataframe: pd.DataFrame) -> tuple:
    """
    Split the dataset into
    training and testing subsets.
    """

    X = dataframe.drop(columns=["label"])

    y = dataframe["label"]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

# =========================================================
# Model training
# =========================================================

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train the Random Forest classifier.
    """

    logger.info("Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train
    )

    logger.info("Model trained successfully.")

    return model

# =========================================================
# Model evaluation
# =========================================================

def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Evaluate the trained classifier.
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    logger.info(f"Accuracy: {accuracy:.4f}")

    report = classification_report(
        y_test,
        predictions
    )

    logger.info("\nClassification Report\n")

    logger.info(f"\n{report}")

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    logger.info("Confusion Matrix")

    logger.info(f"\n{matrix}")

# =========================================================
# Main workflow
# =========================================================

def main():
    """
    Execute the supervised learning pipeline.
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

    datasets: list[pd.DataFrame] = []

    for file_path in dataset_files:

        dataframe = load_feature_dataset(file_path)

        dataframe = assign_label(
            dataframe,
            file_path
        )

        datasets.append(dataframe)

    merged_dataset = merge_datasets(datasets)

    X_train, X_test, y_train, y_test = split_dataset(merged_dataset)

    logger.info(f"Training samples: {len(X_train)}")

    logger.info(f"Testing samples : {len(X_test)}")

    print_separator(logger, "-")

    model = train_random_forest(
        X_train,
        y_train
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )

    print_separator(logger, "-")

    save_model(
        model,
        MODELS_DIR / "random_forest_model.joblib",
        logger
    )

    execution_time = stop_timer(start_time)

    print_summary(
        logger,
        "PROCESS SUMMARY",
        {
            "Datasets": len(dataset_files),
            "Samples": len(merged_dataset),
            "Train": len(X_train),
            "Test": len(X_test),
            "Execution": f"{execution_time:.2f} s",
            "Output": MODELS_DIR
        }
    )

# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()