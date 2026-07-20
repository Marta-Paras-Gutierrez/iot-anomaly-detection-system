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

# Model serialization
import joblib

# Data manipulation
import pandas as pd

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
    start_timer,
    stop_timer,
    print_separator,
    print_summary
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

def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load one feature dataset.
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

    if "normal" in filename:
        dataframe["label"] = 0
    else:
        dataframe["label"] = 1

    return dataframe

# =========================================================
# Dataset merging
# =========================================================

def merge_datasets(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge every labelled dataset.
    """

    dataframe = pd.concat(
        datasets,
        ignore_index=True
    )

    logger.info(f"Total samples: {len(dataframe)}")

    return dataframe

# =========================================================
# Prepare evaluation dataset
# =========================================================

def prepare_dataset(dataframe: pd.DataFrame):
    """
    Separate features and labels.
    """

    X = dataframe.drop(columns=["label"])

    y = dataframe["label"]

    return X, y

# =========================================================
# Model loading
# =========================================================

def load_models():
    """
    Load every trained model.
    """

    random_forest = joblib.load(MODELS_DIR / "random_forest_model.joblib")

    isolation_forest = joblib.load(MODELS_DIR / "isolation_forest_model.joblib")

    logger.info("Random Forest loaded.")
    logger.info("Isolation Forest loaded.")

    return random_forest, isolation_forest

# =========================================================
# Model evaluation
# =========================================================

def evaluate_model(model, X: pd.DataFrame, y: pd.Series, model_name: str, prediction_transform=None) -> dict:
    """
    Evaluate a trained Machine Learning model.
    """

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )

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
# Save evaluation results
# =========================================================

def save_results(results: list[dict]) -> Path:
    """
    Save evaluation metrics.
    """

    output_file = RESULTS_DIR / "model_comparison.csv"

    dataframe = (
        pd.DataFrame(results).sort_values(
            by="F1-Score",
            ascending=False
        )
    )

    dataframe.to_csv(
        output_file,
        index=False,
        sep=CSV_SEPARATOR,
        encoding=CSV_ENCODING
    )

    logger.info(f"Results saved: {output_file.name}")

    return output_file

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

    datasets = []

    for file_path in dataset_files:
        dataframe = load_dataset(file_path)

        dataframe = assign_label(
            dataframe,
            file_path
        )

        datasets.append(dataframe)

    merged_dataset = merge_datasets(datasets)

    X, y = prepare_dataset(merged_dataset)

    random_forest, isolation_forest = load_models()

    results = []

    rf_results = evaluate_model(
        random_forest,
        X,
        y,
        "Random Forest"
    )

    results.append(rf_results)

    logger.info("")
    logger.info("Random Forest")
    logger.info(f"Accuracy  : {rf_results['Accuracy']:.4f}")
    logger.info(f"Precision : {rf_results['Precision']:.4f}")
    logger.info(f"Recall    : {rf_results['Recall']:.4f}")
    logger.info(f"F1-Score  : {rf_results['F1-Score']:.4f}")
    logger.info("")

    if_results = evaluate_model(
        isolation_forest,
        X,
        y,
        "Isolation Forest",
        lambda p: pd.Series(p).replace({1: 0, -1: 1})
    )

    results.append(if_results)

    logger.info("Isolation Forest")
    logger.info(f"Accuracy  : {if_results['Accuracy']:.4f}")
    logger.info(f"Precision : {if_results['Precision']:.4f}")
    logger.info(f"Recall    : {if_results['Recall']:.4f}")
    logger.info(f"F1-Score  : {if_results['F1-Score']:.4f}")
    logger.info("")

    save_results(results)

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