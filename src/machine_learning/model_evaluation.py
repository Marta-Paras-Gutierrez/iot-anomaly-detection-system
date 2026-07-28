"""
=========================================================
IoT Traffic Monitoring & Anomaly Detection System
=========================================================

Module: 03 - Model Evaluation

Description:

This module evaluates the trained Machine Learning
models using the generated feature datasets.

Both supervised and unsupervised approaches are
tested over the same traffic samples in order to
compare their anomaly detection performance.

Workflow:

1. Load feature datasets
2. Assign labels
3. Merge datasets
4. Load trained models
5. Prepare evaluation dataset

Input: data/datasets/
       models/

Output: results/

Author: Marta Parás
"""

# =========================================================
# Imports
# =========================================================

# File management
from pathlib import Path

# Callable type hints
from collections.abc import Callable

# Data manipulation
import pandas as pd

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Project configuration
from src.config import (
    PROJECT_NAME,
    DATASETS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    CSV_SEPARATOR,
    CSV_ENCODING
)

# Utilities
from src.utils import (
    get_base_filename,
    get_logger,
    load_dataframe,
    load_model,
    start_timer,
    stop_timer,
    print_separator,
    print_summary,
    save_csv
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
    Search every generated feature dataset.
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

    Normal traffic : 0

    Anomalous traffic : 1
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
# Prepare evaluation dataset
# =========================================================

def prepare_dataset(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and labels for evaluation.
    """

    X = dataframe.drop(columns=["label"])

    y = dataframe["label"]

    return X, y

# =========================================================
# Model loading
# =========================================================

def load_models() -> tuple[RandomForestClassifier, IsolationForest]:
    """
    Load every trained model.
    """

    random_forest = load_model(
        MODELS_DIR / "random_forest_model.joblib",
        logger
    )

    isolation_forest = load_model(
        MODELS_DIR / "isolation_forest_model.joblib",
        logger
    )

    return random_forest, isolation_forest

# =========================================================
# Model evaluation
# =========================================================

def evaluate_model(model, X: pd.DataFrame, y: pd.Series, model_name: str, prediction_transform:  Callable[[pd.Series], pd.Series] | None = None) -> dict[str, float | str]:
    """
    Evaluate a trained Machine Learning model
    using the provided evaluation dataset.
    """

    predictions = model.predict(X)

    if prediction_transform is not None:
        predictions = prediction_transform(predictions)

    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(y, predictions),
        "Recall": recall_score(y, predictions),
        "F1-Score": f1_score(y, predictions)
    }

    logger.info(f"{model_name} evaluated.")

    return results

# =========================================================
# Display metrics
# =========================================================

def log_metrics(results: dict[str, float | str]) -> None:
    """
    Log the evaluation metrics of one model.
    """

    logger.info(results["Model"])
    logger.info(f"Accuracy  : {results['Accuracy']:.4f}")
    logger.info(f"Precision : {results['Precision']:.4f}")
    logger.info(f"Recall    : {results['Recall']:.4f}")
    logger.info(f"F1-Score  : {results['F1-Score']:.4f}")
    logger.info("")

# =========================================================
# Main workflow
# =========================================================

def main():
    """
    Execute the model evaluation pipeline.
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

    X, y = prepare_dataset(merged_dataset)

    random_forest, isolation_forest = load_models()

    results: list[dict] = []

    rf_results = evaluate_model(
        random_forest,
        X,
        y,
        "Random Forest"
    )

    results.append(rf_results)
    log_metrics(rf_results)

    if_results = evaluate_model(
        isolation_forest,
        X,
        y,
        "Isolation Forest",
        lambda predictions: pd.Series(predictions).replace({1: 0, -1: 1})
    )

    results.append(if_results)
    log_metrics(if_results)

    print_separator(logger, "-")

    comparison = pd.DataFrame(results).sort_values(
        by="F1-Score",
        ascending=False
    )

    save_csv(
        comparison,
        RESULTS_DIR / "model_comparison.csv",
        logger,
        CSV_SEPARATOR,
        CSV_ENCODING
    )

    best_model = max(
        results,
        key=lambda model: model["F1-Score"]
    )

    logger.info(f"Best model : {best_model['Model']}")

    logger.info(f"Best F1-Score : {best_model['F1-Score']:.4f}")

    execution_time = stop_timer(start_time)

    print_summary(
        logger,
        "PROCESS SUMMARY",
        {
            "Models": len(results),
            "Samples": len(X),
            "Execution": f"{execution_time:.2f} s",
            "Output": RESULTS_DIR
        }
    )

# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()