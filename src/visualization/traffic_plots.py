"""
=========================================================
IoT Traffic Monitoring & Anomaly Detection System
=========================================================

Module: Traffic Visualizations

Description:

This module generates graphical visualizations from the
generated datasets and Machine Learning results.

The figures are automatically saved and can later be
used inside the dashboard or the project report.

Workflow:

1. Load datasets
2. Generate visualizations
3. Save figures

Input:
    data/
    results/

Output:
    images/visualizations/

Author: Marta Parás
"""

# =========================================================
# Imports
# =========================================================

# File management
from pathlib import Path

# Data visualization
import matplotlib.pyplot as plt

# Data manipulation
import pandas as pd

# Machine Learning
from sklearn.metrics import ConfusionMatrixDisplay

# Project configuration
from src.config import (
    PROJECT_NAME,
    DETECTION_RESULTS_DIR,
    MODELS_DIR,
    DATASETS_DIR,
    RESULTS_DIR,
    VISUALIZATION_DIR,
    CSV_SEPARATOR,
    CSV_ENCODING
)

# Utilities
from src.utils import (
    get_logger,
    load_dataframe,
    load_model,
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
# Dataset loading
# =========================================================

def load_visualization_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load one CSV dataset.
    """

    logger.info(f"Loading: {file_path.name}")

    return load_dataframe(
        file_path,
        separator=CSV_SEPARATOR,
        encoding=CSV_ENCODING
    )

# =========================================================
# Figure saving
# =========================================================

def save_figure(filename: str) -> None:
    """
    Save current matplotlib figure.
    """

    output_file = VISUALIZATION_DIR / filename

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logger.info(f"Figure saved: {filename}")

# =========================================================
# Model comparison loading
# =========================================================

def load_model_results() -> pd.DataFrame:
    """
    Load model comparison results.
    """

    dataframe = load_visualization_dataset(RESULTS_DIR / "model_comparison.csv")

    logger.info("Model comparison loaded.")

    return dataframe

# =========================================================
# Plot model comparison
# =========================================================

def plot_model_comparison(dataframe: pd.DataFrame) -> None:
    """
    Plot model comparison.
    """

    metrics = dataframe.set_index("Model")[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score"
        ]
    ]

    ax = metrics.plot(
        kind="bar",
        figsize=(8, 5)
    )

    ax.set_title("Model Performance Comparison")

    ax.set_ylabel("Score")

    ax.set_ylim(0, 1.05)

    ax.grid(axis="y")

    save_figure("model_comparison.png")

# =========================================================
# Anomalies by capture
# =========================================================

def plot_anomalies_by_capture() -> None:
    """
    Plot anomaly percentage for every analysed capture.
    """

    captures: list[str] = []

    percentages: list[float] = []

    for file_path in sorted(DETECTION_RESULTS_DIR.glob("*_detections.csv")):

        dataframe = load_visualization_dataset(file_path)

        total = len(dataframe)

        anomalies = dataframe["final_prediction"].sum()

        percentage = (anomalies / total) * 100

        captures.append(file_path.stem.replace("_detections", ""))

        percentages.append(percentage)

    figure_height = max(5, len(captures) * 0.6)

    plt.figure(figsize=(10, figure_height))

    bars = plt.barh(
        captures,
        percentages
    )

    plt.xlim(0, 100)

    plt.xlabel("Detected Anomalies (%)")

    plt.title("Anomaly Percentage by Capture")

    plt.grid(axis="x")

    for bar, value in zip(bars, percentages):
        plt.text(
            value + 1,
            bar.get_y() + bar.get_height()/2,
            f"{value:.1f}%",
            va="center"
        )

    save_figure("anomalies_by_capture.png")

# =========================================================
# Confusion matrix
# =========================================================

def plot_confusion_matrix() -> None:
    """
    Plot the Random Forest confusion matrix.
    """

    model = load_model(
        MODELS_DIR / "random_forest_model.joblib",
        logger
    )

    datasets: list[pd.DataFrame] = []

    for file_path in sorted(DATASETS_DIR.glob("*_dataset.csv")):
        dataframe = load_visualization_dataset(file_path)

        label = 0 if "normal" in file_path.stem.lower() else 1

        dataframe["label"] = label

        datasets.append(dataframe)

    dataframe = pd.concat(
        datasets,
        ignore_index=True
    )

    X = dataframe.drop(columns=["label"])

    y = dataframe["label"]

    predictions = model.predict(X)

    display = ConfusionMatrixDisplay.from_predictions(
        y,
        predictions,
        cmap="Blues"
    )

    display.ax_.set_title("Random Forest Confusion Matrix")

    save_figure("confusion_matrix.png")

# =========================================================
# Main workflow
# =========================================================

def main() -> None:
    """
    Generate project figures.
    """

    start_time = start_timer()

    print_separator(logger)
    logger.info(PROJECT_NAME)
    print_separator(logger)

    metrics = load_model_results()

    plot_model_comparison(metrics)

    plot_anomalies_by_capture()

    plot_confusion_matrix()

    execution_time = stop_timer(start_time)

    print_summary(
        logger,
        "PROCESS SUMMARY",
        {
            "Figures": 3,
            "Execution": f"{execution_time:.2f} s",
            "Output": VISUALIZATION_DIR
        }
    )

# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()