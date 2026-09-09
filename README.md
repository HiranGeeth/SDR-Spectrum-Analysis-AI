# SDR-Based Real-Time Spectrum Analysis & Modulation Recognition

## Hybrid CNN-Random Forest Ensemble for RF Signal Classification

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![RTL-SDR](https://img.shields.io/badge/RTL--SDR-V3-green.svg)](https://www.rtl-sdr.com/)

---

##  Project Overview

An **end-to-end AI-powered RF signal classification system** that uses a **hybrid CNN-Random Forest ensemble** to identify and classify radio signals in real-time using an **RTL-SDR** dongle.

### Key Achievements

| Metric | Value |
|--------|-------|
| **Best Model Accuracy** | 44.18% (CNN on raw IQ) |
| **Ensemble Accuracy** | 44.54% (Average Probabilities) |
| **Real-Time Inference** |  Yes (RTL-SDR) |
| **Spectral Resolution** | 2.4 MHz bandwidth |
| **Modulation Classes** | 11 |
| **Training Samples** | 220,000 (RadioML 2016.10A) |

###  What Makes This Project Unique

- **Hybrid Ensemble**: Combines strengths of Random Forest (traditional ML) and CNN (deep learning) for robust classification
- **Real-Time SDR Integration**: Direct hardware interface with RTL-SDR Blog V3 dongle
- **Interactive Dashboard**: Streamlit-based UI with live spectrum visualization, waterfall plots, and classification results
- **Full Explainability**: Every prediction is traceable through ensemble voting

---
##  Model Performance Comparison

| Model | Architecture | Accuracy | Training Time | Inference Time |
|-------|--------------|----------|---------------|----------------|
| **Random Forest** | 100 trees, depth 10 | 40.72% | 1 min | <10ms |
| **CNN (Raw IQ)** | 4 Conv layers + Dense | 44.18% | 10 min | <50ms |
| **ResNet-50 (Transfer)** | ImageNet pretrained | In progress | 3+ hours | <100ms |
| **Ensemble** | Weighted Average (RF + CNN) | **44.54%** | 11 min | <60ms |

### Ensemble Improvement Analysis

| Metric | Value |
|--------|-------|
| **Both Models Correct** | 35.2% |
| **Both Models Wrong** | 42.1% |
| **RF Only Correct** | 5.5% |
| **CNN Only Correct** | 17.2% |

**Key Insight**: The CNN and Random Forest make **different types of errors**, making the ensemble more robust than either individual model.

---

##  Technical Stack

| Component | Technology |
|-----------|------------|
| **RF Hardware** | RTL-SDR Blog V3 (Rafael Micro R820T) |
| **SDR Control** | pyrtlsdr (v0.4.0) |
| **Deep Learning** | TensorFlow 2.13+ / Keras |
| **Classical ML** | Scikit-learn (Random Forest) |
| **Signal Processing** | NumPy, SciPy, Scipy.signal |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **UI Framework** | Streamlit 1.28+ |
| **Python Version** | 3.11+ |

---

---

##  Quick Start

### Prerequisites

- **Hardware**: RTL-SDR Blog V3 dongle
- **Software**: Python 3.11+, Windows/Linux/Mac
- **Drivers**: WinUSB (via Zadig) for RTL-SDR on Windows

### Installation

# Clone repository
git clone https://github.com/yourusername/SDR-Spectrum-Analysis-AI.git
cd SDR-Spectrum-Analysis-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download Dataset
RadioML 2016.10A

# Train Random Forest
python src/train_rf.py

# Train CNN
python src/train_cnn.py

# Run Ensemble
python src/ensemble_predict.py

# Connect RTL-SDR and launch dashboard
streamlit run src/live_scanner.py

## Contributing

PRs welcome! Focus areas:

- Additional signal classes
- Model architecture improvements
- UI enhancements
- Documentation

---

## License

MIT License

---

## Acknowledgments

- [RadioML Dataset](https://www.kaggle.com/datasets/nolasthitnotomorrow/radioml2016-deepsigcom) by DeepSig
- [RTL-SDR Blog](https://www.rtl-sdr.com/) for hardware and drivers
- [pyrtlsdr](https://github.com/roger-/pyrtlsdr) Python wrapper

---

##  Contact

**Author**: Hiran Dharmapala  
**LinkedIn**: [[Hiran Dharmapala](https://www.linkedin.com/in/hiran-dharmapala-12c/)]  
**Project Link**: [https://github.com/HiranGeeth/SDR-Spectrum-Analysis-AI](https://github.com/HiranGeeth/SDR-Spectrum-Analysis-AI)

---

##  If this project helped you, star it on GitHub!

---

*Built with Python, TensorFlow, Scikit-learn, and RTL-SDR* 
