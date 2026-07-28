"""
=========================================================
IoT Traffic Monitoring & Anomaly Detection System
=========================================================

Module: Traffic Detection

Description:

This module analyses new network captures using the
previously trained Machine Learning models.

Each capture is processed following the same pipeline
used during training in order to generate the required
traffic features before performing anomaly detection.

Workflow:

1. Search detection captures
2. Convert capture into DataFrame
3. Clean DataFrame
4. Extract traffic features
5. Load trained models
6. Predict anomalies
7. Save detection results

Input:
    data/detection/

Output:
    results/detection/

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
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# Project configuration
from src.config import (
    PROJECT_NAME,
    DETECTION_DATA_DIR,
    DETECTION_RESULTS_DIR,
    MODELS_DIR,
    SUPPORTED_CAPTURE_FORMATS,
    CSV_SEPARATOR,
    CSV_ENCODING
)

# Utilities
from src.utils import (
    get_logger,
    get_capture_files,
    get_base_filename,
    load_model,
    save_csv,
    start_timer,
    stop_timer,
    print_separator,
    print_summary
)

# Packet extraction
from src.data_processing.convert_pcap_to_dataset import convert_capture_to_dataframe

# Data cleaning
from src.data_processing.preprocess_dataframe import (
    validate_dataframe,
    clean_dataframe
)

# Feature extraction
from src.data_processing.extract_features import (
    create_time_windows,
    extract_features
)

# =========================================================
# Logger configuration
# =========================================================

logger = get_logger(__name__)

# =========================================================
# Detection capture discovery
# =========================================================

def get_detection_files() -> list[Path]:
    """
    Search every packet capture available
    for anomaly detection.
    """

    return get_capture_files(
        DETECTION_DATA_DIR,
        SUPPORTED_CAPTURE_FORMATS
    )

# =========================================================
# Model loading
# =========================================================

def load_models() -> tuple[RandomForestClassifier, IsolationForest]:
    """
    Load every trained Machine Learning model.
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
# Feature generation
# =========================================================

def generate_feature_dataset(capture_path: Path) -> pd.DataFrame:
    """
    Generate the feature dataset associated
    with one packet capture.
    """

    logger.info(f"Processing capture: {capture_path.name}")

    dataframe = convert_capture_to_dataframe(capture_path)

    validate_dataframe(dataframe)

    dataframe = clean_dataframe(dataframe)

    grouped_packets = create_time_windows(dataframe)

    feature_dataframe = extract_features(grouped_packets)

    logger.info(f"Generated windows: {len(feature_dataframe)}")

    return feature_dataframe

# =========================================================
# Prediction
# =========================================================

def predict_anomalies(random_forest: RandomForestClassifier, isolation_forest: IsolationForest, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Predict anomalies using both
    Machine Learning models.
    """

    rf_predictions = random_forest.predict(dataframe)

    if_predictions = isolation_forest.predict(dataframe)

    if_predictions = pd.Series(if_predictions).replace({1: 0, -1: 1})

    results = dataframe.copy()

    results["random_forest"] = rf_predictions

    results["isolation_forest"] = if_predictions

    results["final_prediction"] = ((rf_predictions == 1) | (if_predictions == 1)).astype(int)

    logger.info("Predictions completed.")

    return results

# =========================================================
# Save detection results
# =========================================================

def save_detection_results(dataframe: pd.DataFrame, capture_path: Path) -> Path:
    """
    Save anomaly detection results.
    """

    output_file = DETECTION_RESULTS_DIR / (f"{get_base_filename(capture_path)}_detections.csv")

    return save_csv(
        dataframe,
        output_file,
        logger,
        CSV_SEPARATOR,
        CSV_ENCODING
    )

# =========================================================
# Detection summary
# =========================================================

def print_detection_summary(dataframe: pd.DataFrame) -> None:
    """
    Display anomaly detection statistics.
    """

    total_windows = len(dataframe)

    anomalies = dataframe["final_prediction"].sum()

    normal = total_windows - anomalies

    logger.info("\nDetection summary")
    logger.info(f"Windows analysed : {total_windows}")
    logger.info(f"Normal windows   : {normal}")
    logger.info(f"Anomalies found  : {anomalies}")
    logger.info("")

# =========================================================
# Main workflow
# =========================================================

def main():
    """
    Execute the anomaly detection pipeline.
    """

    start_time = start_timer()

    print_separator(logger)
    logger.info(PROJECT_NAME)
    print_separator(logger)

    capture_files = get_detection_files()

    if not capture_files:
        logger.warning("No detection captures were found.")
        return

    logger.info(f"Captures detected: {len(capture_files)}\n")

    random_forest, isolation_forest = load_models()

    successful = 0

    for capture_path in capture_files:

        print_separator(logger, "-")

        feature_dataframe = generate_feature_dataset(capture_path)

        results = predict_anomalies(
            random_forest,
            isolation_forest,
            feature_dataframe
        )

        save_detection_results(
            results,
            capture_path
        )

        print_detection_summary(results)

        successful += 1

    execution_time = stop_timer(start_time)

    print_summary(
        logger,
        "PROCESS SUMMARY",
        {
            "Captures": successful,
            "Execution": f"{execution_time:.2f} s",
            "Output": DETECTION_RESULTS_DIR
        }
    )

# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()