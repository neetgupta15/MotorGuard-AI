# 🔧 Electric Motor Fault Detection System
## Predictive Maintenance using Vibration Analysis & Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![STM32](https://img.shields.io/badge/STM32-H743ZI2-03234B?style=flat-square&logo=stmicroelectronics&logoColor=white)](https://www.st.com)
[![MATLAB](https://img.shields.io/badge/MATLAB-Simulink-0076A8?style=flat-square)](https://www.mathworks.com)

---

## 📋 Project Overview

This project implements a **model-based predictive maintenance system** for electric motors using vibration analysis. It employs a complete pipeline from **synthetic data generation** through **ML-based fault classification** to **embedded deployment** on an STM32 microcontroller.

### Key Features
- 🏭 **Physics-based vibration simulator** generating realistic 3-axis accelerometer data
- 🔍 **96-feature extraction** across time, frequency, and wavelet domains
- 🤖 **Multiple ML models** (Random Forest, SVM, XGBoost, DNN) with automated comparison
- 📡 **STM32 embedded firmware** for real-time fault detection with LIS3DH accelerometer
- 📊 **Real-time monitoring dashboard** with stunning dark glassmorphism UI
- 📝 **MATLAB/Simulink reference scripts** for academic complement

### Fault Classes Detected

| # | Fault Type | Detection Method |
|---|-----------|-----------------|
| 0 | Normal Operation | Baseline vibration signature |
| 1 | Bearing Fault | Impulse trains at BPFO/BPFI frequencies |
| 2 | Rotor Imbalance | Dominant 1× shaft frequency |
| 3 | Shaft Misalignment | Strong 2× component + axial vibration |
| 4 | Electrical Fault | 2× line frequency sidebands |

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Data Generation │ →  │ Preprocessing │ →  │ Feature Extract │
│  (Simulator)     │    │ (Filter/Norm) │    │ (96 features)   │
└─────────────────┘    └──────────────┘    └────────┬────────┘
                                                     │
                     ┌──────────────────────────────┘
                     ▼
          ┌────────────────────┐     ┌─────────────────┐
          │  ML Model Training │ →   │    Validation    │
          │  (RF/SVM/DNN)      │     │  (End-to-End)    │
          └────────┬───────────┘     └─────────────────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
┌──────────────┐    ┌─────────────────┐
│ STM32 Deploy │    │  Web Dashboard  │
│ (Firmware)   │    │  (Monitoring)   │
└──────────────┘    └─────────────────┘
```

---

## 📁 Project Structure

```
Fault Detection/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── src/                               # Python source code
│   ├── data_generation/
│   │   ├── motor_simulator.py         # Physics-based vibration simulator
│   │   ├── bearing_model.py           # Bearing fault frequency model
│   │   └── generate_dataset.py        # Batch dataset generator
│   ├── preprocessing/
│   │   └── preprocessing.py           # Filter, normalize, segment
│   ├── features/
│   │   ├── feature_extractor.py       # 96-feature extraction engine
│   │   └── feature_analysis.py        # Feature importance analysis
│   ├── models/
│   │   ├── train_models.py            # Train RF, SVM, XGBoost, DNN
│   │   ├── evaluate_models.py         # Confusion matrix, ROC curves
│   │   └── export_model.py            # TFLite export for STM32
│   └── validation/
│       └── validate_system.py         # End-to-end validation
├── stm32_firmware/                    # Embedded C firmware
│   ├── Inc/
│   │   ├── main.h                     # System configuration
│   │   ├── lis3dh.h                   # Accelerometer driver header
│   │   └── fault_detector.h           # Detection module header
│   └── Src/
│       ├── main.c                     # Main application loop
│       ├── lis3dh_driver.c            # LIS3DH I2C driver
│       └── fault_detector.c           # Feature extraction + classify
├── dashboard/                         # Web monitoring interface
│   ├── index.html                     # Dashboard structure
│   ├── style.css                      # Premium dark theme
│   └── dashboard.js                   # Real-time visualization engine
├── matlab/                            # MATLAB/Simulink reference
│   ├── motor_simulink_model.m         # Motor model & vibration gen
│   └── feature_extraction_matlab.m    # PMT feature extraction
├── docs/                              # Documentation
│   ├── PROJECT_REPORT.md              # Academic project report
│   └── HARDWARE_SETUP.md              # Hardware wiring guide
└── data/                              # Generated data (not in repo)
    ├── raw/                           # Raw vibration CSVs
    ├── processed/                     # Preprocessed numpy arrays
    └── features/                      # Extracted feature matrices
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset

```bash
python -m src.data_generation.generate_dataset --samples 200
```

### 3. Preprocess Data

```bash
python -m src.preprocessing.preprocessing
```

### 4. Extract Features

```bash
python -m src.features.feature_extractor
```

### 5. Train ML Models

```bash
python -m src.models.train_models
```

### 6. Evaluate & Visualize

```bash
python -m src.models.evaluate_models
```

### 7. Validate System

```bash
python -m src.validation.validate_system
```

### 8. Open Dashboard

Open `dashboard/index.html` in a web browser to see the real-time monitoring interface.

---

## 🖥️ Web Dashboard

The monitoring dashboard provides real-time visualization of:
- **Motor Health Score** — Animated ring gauge
- **3-Axis Vibration Waveforms** — Live updating chart
- **FFT Spectrum Analysis** — Frequency domain visualization
- **Fault Classification** — Real-time prediction with confidence
- **Alert System** — Timestamped fault notifications
- **Health Trend** — Historical health score tracking
- **Detection Timeline** — Color-coded fault history

---

## 🔩 Hardware Deployment

### Required Components
- STM32 NUCLEO-H743ZI2 development board
- LIS3DH 3-axis accelerometer breakout board
- Jumper wires, buzzer, LEDs
- Electric motor (DC/BLDC) for testing

### Wiring (I2C Connection)
| LIS3DH Pin | STM32 Pin | Function |
|-----------|-----------|----------|
| VCC | 3.3V | Power |
| GND | GND | Ground |
| SDA | PB9 | I2C1 Data |
| SCL | PB8 | I2C1 Clock |
| INT1 | PA1 | Data Ready |

See `docs/HARDWARE_SETUP.md` for detailed wiring diagrams.

---

## 📊 Results

The system achieves high classification accuracy across all fault types:

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Random Forest | ~95% | ~0.95 |
| SVM (RBF) | ~93% | ~0.93 |
| Gradient Boosting | ~96% | ~0.96 |
| Dense Neural Network | ~94% | ~0.94 |

---

## 🎓 Expertise Gained

- Artificial Intelligence & Machine Learning
- Signal Processing & Vibration Analysis
- Embedded Systems (ARM Cortex-M7)
- Model-Based Design (Simulink)
- Predictive Maintenance & Health Monitoring
- Real-Time Data Processing
- Web Dashboard Development

---

## 📚 References

1. Randall, R.B. "Vibration-based Condition Monitoring"
2. Harris, T.A. "Rolling Bearing Analysis", 5th Edition
3. IEC 60034-14: Vibration standards for rotating machines
4. MathWorks — "Detect and Diagnose Faults"
5. STMicroelectronics — LIS3DH Datasheet & AN3308
6. TensorFlow Lite for Microcontrollers Documentation

---

## 📄 License

This project is developed for academic purposes as part of a predictive maintenance research initiative.
