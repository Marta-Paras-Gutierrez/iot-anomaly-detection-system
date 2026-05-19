# Traffic Analysis Workflow

## Overview

This document describes the initial workflow used for IoT traffic analysis and anomaly detection within the project.

The objective of this phase is to create a reproducible pipeline capable of processing captured network traffic and preparing datasets for Machine Learning models.

---

## Workflow Stages

### 1. IoT Network Simulation

IoT environments are simulated using GNS3 to reproduce realistic domestic and industrial network scenarios.

The simulation environment allows controlled generation of:
- Normal traffic
- Abnormal traffic
- Connectivity anomalies
- Simulated reconnection events

---

### 2. Traffic Capture

Network traffic is captured using:
- Wireshark
- Tshark
- PCAP files

Captured traffic includes:
- Packet timestamps
- Source and destination addresses
- Protocol information
- Packet size and frequency

---

### 3. Data Preprocessing

The preprocessing pipeline includes:
- Packet filtering
- Data cleaning
- Traffic normalization
- Feature extraction
- Dataset structuring

The processed data is prepared for Machine Learning analysis.

---

### 4. Feature Engineering

Relevant traffic features are extracted, including:
- Packet frequency
- Traffic volume
- Communication periodicity
- Connection behaviour
- Protocol usage patterns

These features help identify anomalous behaviours in IoT environments.

---

### 5. Machine Learning Analysis

The project evaluates:
- Supervised approaches
- Unsupervised anomaly detection techniques

Current models include:
- Random Forest
- Isolation Forest

---

### 6. Visualization

Results are visualized using Streamlit dashboards to improve:
- Traffic monitoring
- Pattern analysis
- Anomaly interpretation
- System usability

---

## Current Status

The workflow architecture and traffic simulation environments have already been completed.

The current development phase focuses on:
- Feature engineering
- Dataset refinement
- Model evaluation
- Dashboard integration

---
