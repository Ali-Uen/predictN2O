# predictN2O: N₂O Emission Forecasting Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python)
![ML Pipeline](https://img.shields.io/badge/ML_Pipeline-Research_Ready-success?style=for-the-badge)
![Research](https://img.shields.io/badge/Master_Thesis-TH_Köln_2025-blue?style=for-the-badge)


</div>

> **📚 Master's Thesis Technical Appendix**  
> *"Data-Driven Forecasting of Nitrous Oxide Emissions at a Wastewater Treatment Plant"*  
> **Ali Ünal** | TH Köln, 2025

**🚀 Quick Start Guide for Users and Researchers**

---

## 🎯 What is predictN2O?

A **research-ready ML pipeline** for predicting nitrous oxide (N₂O) emissions in wastewater treatment plants using real process data. Built for environmental research with full reproducibility and scientific rigor.

**🔗 For system architecture and technical details:** See [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 📋 Quick Navigation

- [🔧 Setup & Installation](#setup--installation)
- [🚀 Usage Examples](#usage-examples)  
- [📁 Repository Structure](#repository-structure)
- [📓 Jupyter Notebooks](#jupyter-notebooks)
- [🔬 Reproducibility](#reproducibility-notes)
- [📄 License & Contact](#license--contact)

---

## 🔧 Setup & Installation

**Requirements:** Python ≥ 3.8, TensorFlow (for DNN support)

```bash
# 1. Clone repository
git clone <repository-url>
cd predictN2O

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage Examples

### Basic Pipeline Execution
```bash
# Run with default settings
python -m src.main

# Custom model and parameters
python -m src.main --model XGBoost --split 0.7 --augment 3 --noise 0.01
```

### Web Interface
```bash
# Launch interactive dashboard
streamlit run web_app.py
```

**📊 Available Models:** RandomForest, XGBoost, KNN, DecisionTree, AdaBoost, DNN  
**⚙️ Key Parameters:**
- `--model` Model type
- `--split` Train/test ratio (default: 0.8)
- `--augment` Data augmentation rounds  
- `--noise` Augmentation noise level

**📁 Outputs:** All results saved to `/results/` directory

---

## 📁 Repository Structure

```text
predictN2O/
├── 📊 data/                    # Altenrhein WWTP dataset (2016)
├── 📓 notebooks/               # Jupyter analysis notebooks  
├── 📈 results/                 # Generated models & visualizations
├── 🔧 src/                     # Core pipeline source code
│   ├── 🧠 core/               # Modern architecture components
│   │   ├── config_manager.py   # YAML configuration system
│   │   ├── plugin_system.py    # Feature engineering plugins
│   │   └── dataset_analyzer.py # Auto-configuration
│   ├── 🔌 plugins/            # Extensible plugin ecosystem
│   └── 📋 [other modules]     # Data loading, models, evaluation
├── ⚙️ config/                 # YAML configuration files
└── 📋 requirements.txt        # Python dependencies
```

**🔗 Full architecture details:** [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)



---

## 📓 Jupyter Notebooks

**Interactive research and analysis workflows**

| Notebook | Purpose | Content |
|----------|---------|---------|
| `01_EDA.ipynb` | **Exploratory Data Analysis** | Time series visualization, feature correlations, data distributions |
| `02_Prophet_Training.ipynb` | **Prophet Baseline** | Alternative forecasting approach with Prophet model |
| `03_Results_Visualization.ipynb` | **Results Analysis** | Comprehensive model comparison and performance visualization |

**💡 Usage:** Each notebook is self-contained and can be run independently after basic setup.

---

---

## 🎓 Scientific Context

### 📚 **Master Thesis Integration**
- **Title**: "Data-Driven Forecasting of Nitrous Oxide Emissions at a Wastewater Treatment Plant"
- **Author**: Ali Ünal
- **Institution**: TH Köln, 2025
- **Domain**: Environmental Data Science & Climate Protection  
- **Repository Role**: Technical appendix to the thesis providing full implementation

### 🔬 **Reproducibility Features**
- **Version Control**: Complete Git history with tagged releases
- **Configuration Management**: YAML-based parameter tracking
- **Deterministic Results**: Fixed random seeds across all components
- **Documentation**: Comprehensive API docs and usage examples
- **Testing**: Unit tests for critical pipeline components

### 📊 **Dataset Information**
- **Source**: Altenrhein WWTP (Switzerland) - ARA Altenrhein AG
- **DOI**: [https://doi.org/10.25678/0003H2](https://doi.org/10.25678/0003H2)
- **Time Frame**: Calendar year 2016 (15-minute intervals)
- **Variables**: N₂O, dissolved oxygen (DO), temperature, flow
- **Usage**: Academic and non-commercial purposes only

### 🌍 **Environmental Impact**
- **Climate Relevance**: N₂O has 270x CO₂ warming potential
- **Challenge**: Traditional mechanistic models struggle with dynamic emission peaks
- **Solution**: ML/DL models as "soft sensors" for real-time monitoring
- **Research Contribution**: Open-source tool for environmental data science


## 📄 License & Contact

**📜 License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

**👤 Author:** Ali Ünal  
**📧 Contact:** ali@unal.de  
**🎓 Institution:** TH Köln, 2025  
**📚 Thesis:** Technical appendix for full implementation reproducibility

**🤝 Contributing:** Issues and pull requests welcome!

