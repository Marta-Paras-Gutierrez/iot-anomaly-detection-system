"""
=========================================================
IoT Traffic Monitoring & Anomaly Detection System
=========================================================

Main application

This module is the main entry point of the project.

From here the complete workflow can be executed,
including preprocessing, Machine Learning training,
traffic detection and visualization.

Workflow:

1. Display interactive menu
2. Execute selected module
3. Launch dashboard
4. Execute complete workflow

Author: Marta Parás
"""

# =========================================================
# Imports
# =========================================================

# Operating system
import os

# System utilities
import sys

# Process execution
import subprocess

# Project configuration
from src.config import PROJECT_NAME

# =========================================================
# Console utilities
# =========================================================

def clear_console() -> None:
    """
    Clear terminal.
    """

    os.system("cls" if os.name == "nt" else "clear")


def pause() -> None:
    """
    Wait for user.
    """

    input("\nPress ENTER to continue...")

# =========================================================
# Header
# =========================================================

def print_header() -> None:

    clear_console()

    print("=" * 65)
    print(f"{PROJECT_NAME:^65}")
    print("=" * 65)

# =========================================================
# Menu
# =========================================================

def print_menu() -> None:

    print()
    print(" DATA PROCESSING")
    print("-" * 40)
    print(" 1. Convert PCAP captures")
    print(" 2. Clean datasets")
    print(" 3. Extract statistical features")

    print()
    print(" MACHINE LEARNING")
    print("-" * 40)
    print(" 4. Train Random Forest")
    print(" 5. Train Isolation Forest")
    print(" 6. Evaluate models")

    print()
    print(" DETECTION")
    print("-" * 40)
    print(" 7. Analyse new captures")

    print()
    print(" VISUALIZATION")
    print("-" * 40)
    print(" 8. Generate figures")
    print(" 9. Launch dashboard")

    print()
    print(" PROJECT")
    print("-" * 40)
    print("10. Execute complete workflow")

    print()
    print(" 0. Exit")
    print()

# =========================================================
# Module execution
# =========================================================

def run_module(module_name: str) -> None:
    """
    Execute a project module.
    """

    print()
    print("-" * 60)
    print(f"Executing: {module_name}")
    print("-" * 60)
    print()

    subprocess.run(
        [sys.executable, "-m", module_name],
        check=True
    )

    pause()

# =========================================================
# Complete workflow
# =========================================================

def run_complete_pipeline() -> None:
    """
    Execute the complete project workflow.
    """

    modules: list[str] = [
        # Data processing
        "src.data_processing.convert_pcap_to_dataset",
        "src.data_processing.preprocess_dataframe",
        "src.data_processing.extract_features",

        # Machine Learning
        "src.machine_learning.supervised_models",
        "src.machine_learning.unsupervised_models",
        "src.machine_learning.model_evaluation",

        # Detection
        "src.detection.detect_traffic",

        # Visualizations
        "src.visualization.traffic_plots"
    ]

    print()
    print("=" * 60)
    print("Executing complete project workflow...")
    print("=" * 60)

    for module in modules:
        print()
        print(f">>> {module}")

        subprocess.run(
            [sys.executable, "-m", module],
            check=True
        )

    print()
    print("=" * 60)
    print("Project completed successfully.")
    print("=" * 60)

    pause()

# =========================================================
# Main application
# =========================================================

def main() -> None:
    """
    Main application.
    """

    while True:

        print_header()

        print_menu()

        option = input("Select an option: ").strip()

        match option:

            # =============================================
            # Exit
            # =============================================

            case "0":
                print("\nClosing application...")
                break

            # =============================================
            # Data processing
            # =============================================

            case "1":
                run_module("src.data_processing.convert_pcap_to_dataset")

            case "2":
                run_module("src.data_processing.preprocess_dataframe")

            case "3":
                run_module("src.data_processing.extract_features")

            # =============================================
            # Machine Learning
            # =============================================

            case "4":
                run_module("src.machine_learning.supervised_models")

            case "5":
                run_module("src.machine_learning.unsupervised_models")

            case "6":
                run_module("src.machine_learning.model_evaluation")

            # =============================================
            # Detection
            # =============================================

            case "7":
                run_module("src.detection.detect_traffic")

            # =============================================
            # Visualization
            # =============================================

            case "8":
                run_module("src.visualization.traffic_plots")

            case "9":
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "streamlit",
                        "run",
                        "src/visualization/dashboard.py"
                    ]
                )

                pause()

            # =============================================
            # Complete workflow
            # =============================================

            case "10":
                run_complete_pipeline()

            # =============================================
            # Invalid option
            # =============================================

            case _:
                print("\nInvalid option.")
                pause()

# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()