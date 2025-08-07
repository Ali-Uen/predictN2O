# PredictN2O: Machine Learning for Nitrous Oxide Emission Forecasting

Nitrous oxide (N₂O) is a potent greenhouse gas, and wastewater treatment plants are significant sources due to biological processes during water purification. This repository provides an open-source, modular pipeline to forecast N₂O emissions from real process data using state-of-the-art machine learning.

PredictN2O combines robust data preparation, feature engineering, model training, and evaluation to deliver reproducible results and enable research in environmental data science.


## Table of Contents

- [Project Overview](#project-overview)
- [Background & Motivation](#background--motivation)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Usage Example](#usage-example)
- [Documentation](#documentation)
- [The Jupyter Notebooks](#the-jupyter-notebooks)
- [License](#license)
- [Contributing & Issues](#contributing--issues)



## Project Overview

PredictN2O is a modular Python pipeline for supervised machine learning (Random Forest, XGBoost, KNN) to forecast N₂O emissions from wastewater treatment data.  
All preprocessing, feature engineering, and evaluation steps are transparent and reproducible.



## Background & Motivation

Wastewater treatment plants release N₂O, a greenhouse gas with a global warming potential over 270 times higher than CO₂.  
Traditional (mechanistic) models struggle to predict dynamic, short-term emission peaks due to the complex, nonlinear nature of the underlying biological processes.  
Machine learning (ML) and deep learning (DL) approaches can process large, real-world datasets and discover hidden patterns, improving the prediction and management of N₂O emissions.  
This project demonstrates how ML/DL models can act as "soft sensors" for real-time environmental monitoring and decision support in wastewater management.


## Repository Structure

The project is structured modularly to ensure high maintainability and reproducibility.

```text
PredictN2O/
│
├── data/                   # Directory for data files.
│   └── (This directory is initially empty. The original dataset is confidential and not included.)
│
├── notebooks/              # Notebooks for data analysis (EDA), model prototyping (Prophet), and final result visualization.
│
├── results/                # All generated outputs (models, figures, logs).
│   ├── model_outputs/      # Saved trained models (.pkl).
│   ├── figures/            # Generated charts and plots.
│   └── logs/               # Log files from pipeline runs.
│
├── src/                    # All Python source code for the pipeline.
│   ├── main.py             # Main script to run the training and evaluation pipeline.
│   ├── config.py           # Central configuration file (paths, periods, etc.).
│   ├── data_loader.py      # Loads and cleans the raw data.
│   ├── feature_engineering.py # Creates new features (lags, rolling features).
│   ├── models.py           # Defines and trains the machine learning models.
│   ├── evaluation.py       # Calculates and displays model performance metrics.
│   ├── augmentation.py     # Augments the training dataset by adding artificial noise.
│   ├── param_grids.py      # Contains hyperparameter grids for model tuning.
│   ├── resampling.py       # Defines resampling rules for each model.
│   └── ...                 # Other utility modules.
│
├── .gitignore              # Specifies files for Git to ignore (e.g., raw data files).
├── LICENSE                 # The project's license file (CC BY-NC-SA 4.0).
├── requirements.txt        # A list of all required Python packages.
└── README.md               # This overview file.
```


## Setup & Installation

1. **Clone the repository**  
   ```sh
   cd PredictN2O
   ```

2. **Install dependencies (ideally in a virtual environment)**
   ```sh
   pip install -r requirements.txt
   ```



## Usage Example

Run the training and evaluation pipeline with default parameters:

```sh
python -m src.main
```

### Parameterization

You can override default parameters via command-line arguments. For example:

```sh
python -m src.main --model XGBoost --split 0.7 --augment 3 --noise 0.01
```
- `--model` Model type (XGBoost, RandomForest, KNN)
- `--split` Train/test split ratio (e.g. 0.8)
- `--augment` Number of data augmentation rounds
- `--noise` Noise level for augmentation

All outputs (trained models, results, and plots) are stored in the `/results/` directory.



## The Jupyter Notebooks

The `/notebooks` directory contains the project's exploratory work, documenting the journey from raw data to the final pipeline. They provide deeper insights into the data and the modeling decisions.

-   **`01_EDA.ipynb` — Exploratory Data Analysis:** This notebook covers the initial investigation of the data. It includes visualizing the time series, analyzing correlations between features, and examining data distributions.

-   **`02_Prophet_Training.ipynb` — Prophet Model Prototyping:** As the Prophet model has a distinct API from the scikit-learn models, its training and evaluation are demonstrated here. It serves as an important baseline and alternative modeling approach.

-   **`03_Results_Visualization.ipynb` — Final Results Visualization:** This notebook loads the predictions saved by the main pipeline and creates comprehensive, often interactive plots to visually compare the performance of different models.



## Documentation

*If you would like a copy of the full Master's Thesis, please contact the author.*



## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.

This means you are free to:
- **Share** — copy and redistribute the material in any medium or format.
- **Adapt** — remix, transform, and build upon the material.

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial** — You may not use the material for commercial purposes.
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

For the full legal code, please see the [LICENSE](LICENSE) file.



## Contributing & Issues

Contributions, bug reports, and feedback are welcome!  
Please open an [issue](https://github.com/ali-unal/PredictN2O/issue) or submit a pull request.

---

**Contact:**  
Ali Ünal — ali@unal.de

