"""
=========================================================
IoT Traffic Monitoring & Anomaly Detection System
=========================================================

Module: 01 - Convert PCAP to DataFrame

Description:

Reads all packet capture files (.pcap / .pcapng) stored in the
project captures directory and converts them into structured
CSV files.

The generated CSV files contain the basic packet information
required by the following stages of the project, where the
traffic will be preprocessed and transformed into Machine
Learning datasets.

Workflow:

1. Search capture files
2. Read packets using PyShark
3. Extract relevant packet fields
4. Build a Pandas DataFrame
5. Export the DataFrame as CSV

Input: captures/raw/

Output: captures/processed/

Author: Marta Parás
"""

# =========================================================
# Imports
# =========================================================

# File and directory management
from pathlib import Path

# Data manipulation
import pandas as pd

# Packet capture processing
import pyshark

# Asynchronous event loop
import asyncio

# Project configuration
from src.config import (
    PROJECT_NAME,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SUPPORTED_CAPTURE_FORMATS,
    CSV_SEPARATOR,
    CSV_ENCODING,
    TSHARK_PATH
)

from src.utils import (
    get_logger,
    start_timer,
    stop_timer,
    print_separator,
    overwrite_file,
    save_dataframe,
    print_summary
)

# =========================================================
# Logger configuration
# =========================================================

logger = get_logger(__name__)

# =========================================================
# Capture discovery
# =========================================================

def get_capture_files() -> list[Path]:
    """
    Search all supported packet capture files.

    Returns
    -------
    list[Path]
        Sorted list containing every capture available
        inside the raw captures directory.
    """

    capture_files = []

    for extension in SUPPORTED_CAPTURE_FORMATS:
        capture_files.extend(RAW_DATA_DIR.glob(f"*{extension}"))

    return sorted(capture_files)

# =========================================================
# Packet extraction
# =========================================================

def extract_packet_information(packet) -> dict:
    """
    Extract the most relevant information from
    a network packet.

    Parameters
    ----------
    packet : pyshark.packet.packet.Packet

    Returns
    -------
    dict
    """

    packet_data = {
        "timestamp": None,
        "packet_length": None,
        "protocol": None,
        "source_ip": None,
        "destination_ip": None,
        "source_port": None,
        "destination_port": None
    }

    # Packet timestamp
    try:
        packet_data["timestamp"] = float(packet.sniff_timestamp)
    except Exception:
        pass

    # Packet size
    try:
        packet_data["packet_length"] = int(packet.length)
    except Exception:
        pass

    # Highest detected protocol
    try:
        packet_data["protocol"] = packet.highest_layer
    except Exception:
        pass

    # IPv4 / IPv6 addresses
    if hasattr(packet, "ip"):
        try:
            packet_data["source_ip"] = packet.ip.src
        except Exception:
            pass

        try:
            packet_data["destination_ip"] = packet.ip.dst
        except Exception:
            pass

    # TCP ports
    if hasattr(packet, "tcp"):
        try:
            packet_data["source_port"] = packet.tcp.srcport
            packet_data["destination_port"] = packet.tcp.dstport
        except Exception:
            pass

    # UDP ports
    elif hasattr(packet, "udp"):
        try:
            packet_data["source_port"] = packet.udp.srcport
            packet_data["destination_port"] = packet.udp.dstport
        except Exception:
            pass

    return packet_data

# =========================================================
# Capture conversion
# =========================================================

def convert_capture_to_dataframe(capture_path: Path) -> pd.DataFrame:
    """
    Convert one capture file into a Pandas DataFrame.

    Parameters
    ----------
    capture_path : Path

    Returns
    -------
    pandas.DataFrame
    """

    logger.info(f"Reading capture: {capture_path.name}")

    # Create an event loop required by PyShark
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    capture = pyshark.FileCapture(
        input_file=str(capture_path),
        tshark_path=TSHARK_PATH,
        keep_packets=False
    )

    packets = []

    for packet in capture:
        packets.append(extract_packet_information(packet))

    capture.close()

    dataframe = pd.DataFrame(packets)

    logger.info(f"Packets extracted: {len(dataframe)}")

    return dataframe

# =========================================================
# Process a single capture
# =========================================================

def process_capture(capture_path: Path) -> int:
    """
    Process an individual packet capture.

    Workflow:
        1. Read capture.
        2. Convert packets into a DataFrame.
        3. Export the DataFrame as CSV.

    Parameters
    ----------
    capture_path : Path
    """

    print_separator(logger, "-")

    logger.info(f"Processing: {capture_path.name}")

    dataframe = convert_capture_to_dataframe(capture_path)

    output_file = (PROCESSED_DATA_DIR / f"{capture_path.stem}_processed.csv")

    overwrite_file(output_file, logger)

    save_dataframe(
        dataframe,
        output_file,
        CSV_SEPARATOR,
        CSV_ENCODING
    )

    logger.info(f"CSV generated: {output_file.name}")

    logger.info("Capture processed successfully.\n")

    return len(dataframe)

# =========================================================
# Main workflow
# =========================================================

def main():
    """
    Execute the complete conversion workflow.
    """
    
    # Start execution timer
    start_time = start_timer()

    print_separator(logger)
    logger.info(PROJECT_NAME)
    print_separator(logger)

    capture_files = get_capture_files()

    # Verify that captures exist
    if not capture_files:
        logger.warning("No capture files were found in the raw directory.")
        return

    logger.info(f"Capture files detected: {len(capture_files)}\n")

    successful = 0
    failed = 0
    total_packets = 0

    # Process every available capture
    for capture in capture_files:
        try:
            packets = process_capture(capture)
            total_packets += packets
            successful += 1
        except Exception as error:
            logger.error(f"Error processing {capture.name}")
            logger.error(error)
            failed += 1

    execution = stop_timer(start_time)

    print_summary(
        logger,
        "PROCESS SUMMARY",
        {
            "Processed": successful,
            "Failed": failed,
            "Packets": total_packets,
            "Execution": f"{execution:.2f} s",
            "Output": PROCESSED_DATA_DIR
        }
    )

# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()