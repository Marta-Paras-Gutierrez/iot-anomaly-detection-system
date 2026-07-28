"""
=========================================================
IoT Traffic Monitoring & Anomaly Detection System
=========================================================

Configuration file

This module centralizes the main project configuration,
including directory paths, processing parameters and
Machine Learning settings.

Author: Marta Parás

=========================================================
"""

# =======================================================
# Imports
# =======================================================

# File and directory management
from pathlib import Path

# =======================================================
# Project information
# =======================================================

PROJECT_NAME = "Intelligent IoT Traffic Monitoring & Anomaly Detection System"

AUTHOR = "Marta Parás"

VERSION = "1.0.0"

# =======================================================
# Project directories
# =======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# =======================================================
# Capture directories
# =======================================================

CAPTURES_DIR = BASE_DIR / "captures"

RAW_DATA_DIR = CAPTURES_DIR / "raw"

PROCESSED_DATA_DIR = CAPTURES_DIR / "processed"

# =======================================================
# Data directories
# =======================================================

DATA_DIR = BASE_DIR / "data"

CLEANED_DATA_DIR = DATA_DIR / "cleaned"

DATASETS_DIR = DATA_DIR / "datasets"

DETECTION_DATA_DIR = BASE_DIR / "data" / "detection"

# =======================================================
# Output directories
# =======================================================

MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR = BASE_DIR / "results"

DETECTION_RESULTS_DIR = BASE_DIR / "results" / "detection"

IMAGES_DIR = BASE_DIR / "images"

VISUALIZATION_DIR = BASE_DIR / "images" / "visualizations"

# =======================================================
# Documentation directories
# =======================================================

NOTEBOOKS_DIR = BASE_DIR / "notebooks"

DOCS_DIR = BASE_DIR / "docs"

# =======================================================
# Capture processing
# =======================================================

# Supported capture formats
SUPPORTED_CAPTURE_FORMATS = [
    ".pcap",
    ".pcapng"
]

# Time window (seconds) used for feature extraction
TIME_WINDOW = 10.0

# =======================================================
# CSV configuration
# =======================================================

CSV_SEPARATOR = ","

CSV_ENCODING = "utf-8"

CSV_EXTENSION = ".csv"

# =======================================================
# Data validation
# =======================================================

# Required columns generated after packet extraction
REQUIRED_COLUMNS = [
    "timestamp",
    "packet_length",
    "protocol",
    "source_ip",
    "destination_ip",
    "source_port",
    "destination_port"
]

# =======================================================
# Machine Learning
# =======================================================

# Random seed used throughout the project
RANDOM_STATE = 42

# Percentage of samples used for testing
TEST_SIZE = 0.30

# Random Forest parameters
N_ESTIMATORS = 100

MAX_DEPTH = None

# =======================================================
# Logging
# =======================================================

LOG_LEVEL = "INFO"

# =======================================================
# Output options
# =======================================================

# Save generated CSV files
SAVE_DATASETS = True

# Save trained models
SAVE_MODELS = True

# Save generated figures
SAVE_FIGURES = True

# Overwrite existing files
OVERWRITE_FILES = False

# =======================================================
# Dashboard
# =======================================================

STREAMLIT_PORT = 8501

# =======================================================
# Wireshark - TShark configuration
# =======================================================

TSHARK_PATH = r"C:\Program Files\Wireshark\tshark.exe"

# =======================================================
# Create required directories
# =======================================================

REQUIRED_DIRECTORIES = [

    # Capture
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,

    # Data
    CLEANED_DATA_DIR,
    DATASETS_DIR,
    DETECTION_DATA_DIR,

    # Models & Results
    MODELS_DIR,
    RESULTS_DIR,
    DETECTION_RESULTS_DIR,

    # Visualization
    VISUALIZATION_DIR,
    IMAGES_DIR,

    # Documentation
    NOTEBOOKS_DIR,
    DOCS_DIR,
]

# Automatically create every required project directory
for directory in REQUIRED_DIRECTORIES:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

