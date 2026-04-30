# Edge-Based Facial Expression Recognition for Mental Health Monitoring

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

## Project Details

*   **Repository Name:** Edge-Based-Facial-Expression-Recognition-for-Mental-Health-Monitoring
*   **Roll Number:** 2210991654
*   **Name:** Himanshi
*   **Project Title:** Edge-Based Facial Expression Recognition for Mental Health Monitoring
*   **Type:** Research
*   **Team Details:** 1 person
*   **Current Status:** Applied

---

## Overview

This project implements an edge-based Facial Expression Recognition (FER) framework tailored for higher education environments. The goal is to provide continuous, privacy-preserving emotional trend monitoring to generate early-warning signals for student welfare interventions.

Unlike cloud-centered AI alternatives that introduce latency and privacy concerns by transmitting sensitive video streams, this framework performs all processing locally on the edge device. It utilizes a lightweight Convolutional Neural Network (CNN), OpenCV-based face processing, and temporal smoothing to output a non-diagnostic "Distress Trend Indicator."

### Key Features
*   **Edge-First Inference:** Runs entirely on local hardware (e.g., workstations, Raspberry Pi) without cloud dependency.
*   **Privacy-Preserving:** Raw video frames never leave the device; only anonymized distress trends are logged.
*   **Lightweight CNN:** Optimized architecture (~1.10 Million parameters) utilizing depthwise-separable convolutions.
*   **Real-Time Performance:** Achieves high throughput and low latency suitable for live webcam streams.
*   **Temporal Smoothing:** Reduces transient misclassifications to calculate a stable distress score over time.

---

## Technical Architecture

The system pipeline consists of the following stages:
1.  **Frame Capture:** RGB video capture via local webcam.
2.  **Face Localization:** OpenCV Haar cascades crop and stabilize facial regions.
3.  **Normalization:** Resizing to 48x48 grayscale tensors with histogram equalization.
4.  **CNN Inference:** Prediction across 7 emotional classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).
5.  **Risk Signaling:** Calculation of a bounded distress indicator aggregated over 5-minute windows.

---

## Installation & Setup

### Prerequisites
*   Python 3.8 or higher
*   A functional webcam

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/Edge-Based-Facial-Expression-Recognition-for-Mental-Health-Monitoring-2210991654-.git](https://github.com/yourusername/Edge-Based-Facial-Expression-Recognition-for-Mental-Health-Monitoring-2210991654-.git)
cd Edge-Based-Facial-Expression-Recognition-for-Mental-Health-Monitoring-2210991654-
