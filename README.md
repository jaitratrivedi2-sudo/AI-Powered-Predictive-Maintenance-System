# 🛠️ AI-Powered Predictive Maintenance System Using Sound & Vibration

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Database](https://img.shields.io/badge/Cloud%20DB-Supabase-3ECF8E.svg)](https://supabase.com/)
[![Alerts](https://img.shields.io/badge/Alerts-Telegram%20Bot-26A5E4.svg)](https://telegram.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

An end-to-end Industrial IoT (IIoT) predictive maintenance platform that detects anomalies, predicts machinery failure, and estimates Remaining Useful Life (RUL) using vibration, sound acoustic signatures, temperature, and pressure sensor data.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Machine Registry & Thresholds](#-machine-registry--thresholds)
- [Prerequisites & Installation](#-prerequisites--installation)
- [Configuration](#-configuration)
- [How to Run](#-how-to-run)
  - [1. Launch Interactive Dashboard](#1-launch-interactive-dashboard)
  - [2. Train Machine Learning Models](#2-train-machine-learning-models)
  - [3. Check System Status & Fleet Health](#3-check-system-status--fleet-health)
  - [4. Sync Historical Data to Supabase](#4-sync-historical-data-to-supabase)
  - [5. Test Cloud Services & Alerts](#5-test-cloud-services--alerts)
- [Hardware Setup (Optional IoT Mode)](#-hardware-setup-optional-iot-mode)
- [Troubleshooting](#-troubleshooting)

---

## 🚀 Overview

Unplanned machine downtime causes massive financial losses in industrial manufacturing. This system provides a 24/7 autonomous monitoring pipeline:
- Ingests telemetry directly from **Arduino / ESP32 hardware** over serial communication.
- Features a **zero-downtime failover engine**: if physical sensors are disconnected or powered off, it seamlessly transitions to **historical CSV simulation** and fallback datasets without crashing.
- Employs **machine learning and signal processing** (FFT spectral peak analysis, RMS, statistical moments) to detect mechanical wear and anomalies.
- Triggers instant **Telegram / Email alerts** to operators before catastrophic failure occurs.
- Visualizes real-time metrics across industrial equipment via an interactive **Streamlit dashboard**.

---

## ✨ Key Features

- **Multi-Sensor Ingestion**:
  - Live serial reading from Arduino/ESP32 (vibration, acoustic sound, temperature, pressure).
  - Multi-tier fallback architecture: `LIVE` ➔ `HISTORICAL` ➔ `FALLBACK`.
- **Advanced Signal Processing & Feature Engineering**:
  - Time-domain statistics: RMS, Peak, Mean, Standard Deviation, Crest Factor.
  - Frequency-domain features: Fast Fourier Transform (FFT) peak frequency tracking.
- **Machine Learning Triad**:
  - **Failure Classification**: Supervised model predicting failure likelihood and risk levels.
  - **Anomaly Detection**: Unsupervised Isolation Forest model capturing abnormal mechanical deviations.
  - **Remaining Useful Life (RUL)**: Regression model calculating estimated operational days remaining.
- **Interactive Monitoring Dashboard**:
  - Fleet-wide status overview with dynamic health scores (0–100%) and condition bands (Excellent, Good, Warning, Critical).
  - Machine-specific drilldown with real-time sensor graphs, FFT spectrum plots, and risk breakdown.
- **Cloud Database & Instant Alerting**:
  - **Supabase Integration**: Cloud storage for sensor telemetry, incident logging, and fleet state.
  - **Instant Alerts**: Automated notifications dispatched via **Telegram Bot** (100% free), Gmail SMTP, or Discord webhook when critical thresholds are breached.

---

## 🏗️ System Architecture

```text
  [ Physical Machine ]
           │
  [ Arduino / ESP32 ]  ──(Serial COM)──┐
  (Sound, Vibration,                   ▼
   Temp, Pressure)             [ Sensor Manager ] ◄── [ Fallback / CSV Data ]
                                       │              (Automatic Failover)
                                       ▼
                          [ Feature Engineering & Preprocessing ]
                            (RMS, FFT Peak, Scaler, Cleaning)
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
   [ Classifier Model ]      [ Isolation Forest ]        [ RUL Estimator ]
   (Failure Probability)     (Anomaly Detection)         (Days Remaining)
            └──────────────────────────┬──────────────────────────┘
                                       │
                                       ▼
                           [ Predictor & Health Engine ]
                           (Health Score: 0 - 100%)
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
  [ Streamlit Dashboard ]                              [ Cloud & Alerting ]
  - Fleet Overview Cards                               - Supabase Cloud DB
  - Telemetry Live Charts                              - Telegram Bot Alerts
  - FFT Frequency Analysis                             - Incident Logging
```

---

## 📁 Repository Structure

```text
Project/
├── app.py                      # Master CLI entry point (dashboard, train, status, cloud sync)
├── config.py                   # Central configuration (paths, thresholds, cloud keys)
├── requirements.txt            # Python dependencies
│
├── arduino_sensor/
│   └── arduino_sensor.ino      # Microcontroller firmware for reading physical sensors
│
├── sensors/
│   ├── sensor_manager.py       # Auto-detects live serial vs fallback data mode
│   ├── sound_reader.py         # Acoustic sensor serial ingestion
│   ├── vibration_reader.py     # Vibration accelerometer reader
│   └── sensor_simulator.py     # Simulated sensor telemetry generator
│
├── preprocessing/
│   ├── data_cleaning.py        # Cleans raw signals & removes missing/extreme values
│   ├── feature_engineering.py  # Computes RMS, FFT peaks, and statistical moments
│   └── scaler.py               # Feature normalization & standard scaler persistence
│
├── models/
│   ├── train_model.py          # Trains Classifier, Anomaly Detector, and RUL models
│   ├── predict_model.py        # Model inference and prediction helpers
│   ├── anomaly_detector.py     # Isolation Forest anomaly detection engine
│   ├── rul_estimator.py        # Remaining Useful Life regression estimator
│   └── saved_models/           # Serialized model artifacts (.pkl files)
│
├── dashboard/
│   ├── dashboard_app.py        # Streamlit interactive monitoring interface
│   └── components/             # Reusable UI widgets (health cards, RUL, charts)
│
├── cloud/
│   ├── database.py             # Supabase client, schema synchronizer, and logging
│   ├── messaging.py            # Telegram Bot, Gmail SMTP, and webhook notifications
│   ├── config_cloud.py         # Cloud services configuration
│   └── supabase_schema.sql     # SQL DDL script for Supabase tables and policies
│
├── utils/
│   ├── alert_system.py         # Threshold checking & event dispatcher
│   ├── data_loader.py          # Robust CSV and telemetry data loading routines
│   ├── logger.py               # Centralized logging configuration
│   ├── model_loader.py         # Safe loader for trained models & scalers
│   └── predictor.py            # Aggregates health score, risk, and anomaly status
│
└── data/
    ├── raw/                    # Raw telemetry data (historical_maintenance.csv)
    ├── processed/              # Engineered features ready for model training
    ├── historical/             # Historical maintenance archives
    └── fallback/               # Failover dataset when sensors are disconnected
```

---

## 🏷️ Machine Registry & Thresholds

The system is configured in `config.py` to monitor various industrial asset types:

| Machine ID | Name | Type | Zone | Normal Vibration | Sound Threshold | Max Temp |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M001** | CNC Machine 01 | CNC | Zone A | < 7.0 mm/s | < 90 dB | 75 °C |
| **M002** | CNC Machine 02 | CNC | Zone A | < 7.0 mm/s | < 90 dB | 75 °C |
| **M003** | Pump Unit 03 | Pump | Zone B | < 6.0 mm/s | < 85 dB | 70 °C |
| **M004** | Pump Unit 04 | Pump | Zone B | < 6.0 mm/s | < 85 dB | 70 °C |
| **M005** | Motor 05 | Motor | Zone C | < 8.0 mm/s | < 95 dB | 80 °C |

### Health Score Bands
- 🟢 **EXCELLENT**: 80% – 100% (Normal operation)
- 🟡 **GOOD**: 60% – 79% (Stable, routine check advised)
- 🟠 **WARNING**: 40% – 59% (Minor anomaly detected, inspect soon)
- 🔴 **CRITICAL**: 0% – 39% (High failure risk / Immediate maintenance required)

---

## ⚙️ Prerequisites & Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd Project
```

### 2. Set up a Python Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv env
.\env\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

All system parameters are defined in `config.py`:

```python
# Hardware / Sensor Settings
SENSOR_AVAILABLE   = False       # Set to True when physical Arduino is connected
SENSOR_PORT        = "COM5"      # Serial port (e.g. COM3 on Windows or /dev/ttyUSB0 on Linux)
SENSOR_BAUD_RATE   = 9600

# Cloud Database & Alerting
SUPABASE_URL       = "https://your-project.supabase.co"
SUPABASE_KEY       = "your-supabase-anon-key"
TELEGRAM_BOT_TOKEN = "your-bot-token"
TELEGRAM_CHAT_ID   = "your-chat-id"

TELEGRAM_ENABLED       = True
CLOUD_DATABASE_ENABLED = True
```

---

## 💻 How to Run

The unified CLI `app.py` gives access to all functions:

### 1. Launch Interactive Dashboard
Run via `app.py`:
```bash
python app.py
```
*Or directly with Streamlit:*
```bash
streamlit run dashboard/dashboard_app.py
```
Open **http://localhost:8501** in your browser to access the live dashboard.

---

### 2. Train Machine Learning Models
Trains the Failure Classifier, Isolation Forest Anomaly Detector, and RUL Regressor, saving `.pkl` artifacts to `models/saved_models/`:
```bash
python app.py --train
```
*Or directly via the training script:*
```bash
python models/train_model.py
```

---

### 3. Check System Status & Fleet Health
Inspects loaded models, active data source mode (`LIVE` vs `HISTORICAL`), and displays current health diagnostics across all registered machines:
```bash
python app.py --status
```

---

### 4. Sync Historical Data to Supabase
Uploads local historical CSV telemetry to your Supabase PostgreSQL cloud table for centralized tracking:
```bash
python app.py --sync-db
```

---

### 5. Test Cloud Services & Alerts
Sends a diagnostic ping to Supabase and dispatches a test alert to your configured Telegram channel:
```bash
python app.py --test-cloud
```

---

## 🔌 Hardware Setup (Optional IoT Mode)

If deploying with physical hardware:
1. Connect an **Arduino / ESP32** to your PC via USB.
2. Wire sensors:
   - **Vibration Sensor** (e.g., ADXL335 / SW-420) to analog input pins.
   - **Acoustic / Sound Sensor** (e.g., KY-038 / LM393) to analog input pins.
   - **Temperature & Pressure Sensors** (optional).
3. Open `arduino_sensor/arduino_sensor.ino` in the Arduino IDE and flash it to your board.
4. Update `config.py`:
   - Set `SENSOR_PORT = "COM5"` (or your board's active COM port).
   - Set `SENSOR_AVAILABLE = True`.
5. Run `python app.py` — live data will now stream in real time.

---

## ❓ Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **"Sensor not connected → loading from historical CSV"** | Physical sensor is unplugged or `SENSOR_AVAILABLE = False`. | Expected behavior! The system automatically utilizes historical fallback data so you can test without hardware. |
| **"Model files not found"** | Models haven't been compiled yet. | Run `python app.py --train` to generate `.pkl` files in `models/saved_models/`. |
| **Telegram messages not sending** | Invalid Bot Token or Chat ID. | Ensure `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are configured correctly in `config.py` and `TELEGRAM_ENABLED = True`. |
| **Port 8501 already in use** | An existing Streamlit instance is running. | Stop previous instances (`Ctrl + C`) or change `DASHBOARD_PORT` in `config.py`. |

---

## 📄 License
This project is licensed under the MIT License - feel free to use, modify, and distribute.

