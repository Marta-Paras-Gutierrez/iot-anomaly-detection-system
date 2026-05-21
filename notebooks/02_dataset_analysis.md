# Dataset Analysis

## Overview

This document describes the structure and characteristics of the datasets generated from the simulated IoT traffic scenarios.

The datasets were created by processing captured network traffic from different domestic and industrial IoT environments.

Both normal and anomalous traffic conditions were analysed to support Machine Learning-based anomaly detection.

---

## Dataset Categories

The current datasets include:

### Normal Traffic
- Domestic IoT traffic
- Industrial IoT traffic

### Anomalous Traffic
- Unreachable destination events
- Simulated reconnection anomalies
- Intermittent DoS activity
- IP duplication scenarios
- Slow exfiltration behaviour
- Irregular communication frequency

---

## Extracted Features

Several traffic-related features were extracted from the PCAP captures, including:

- Total packet count
- Mean packet size
- Packet size deviation
- Inter-arrival times
- Protocol frequency
- ICMP activity
- ARP activity
- Traffic burst patterns
- Temporal communication behaviour

These features allow the system to identify deviations from normal traffic patterns.

---

## Data Processing Workflow

The dataset generation pipeline follows these stages:

1. Traffic capture using Wireshark/TShark
2. PCAP preprocessing
3. Packet filtering
4. Feature extraction
5. CSV dataset generation
6. Dataset normalization and preparation

---

## Current Objectives

The current analysis phase focuses on:

- Comparing normal and anomalous traffic behaviour
- Identifying the most relevant traffic features
- Evaluating feature distributions
- Preparing datasets for ML model training

---

## Planned Analysis

Future analysis stages will include:

- Statistical visualization
- Correlation analysis
- Feature importance evaluation
- Supervised classification
- Unsupervised anomaly detection
- Dashboard integration

---
