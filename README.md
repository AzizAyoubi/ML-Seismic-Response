# Machine Learning Pipeline for Structural Engineering Predictive Modeling

[![License](https://img.shields.io/badge/License-Proprietary-blue.svg)](https://opensource.org/licenses/proprietary)

A comprehensive machine learning pipeline for predicting structural response parameters using various regression algorithms. Designed for seismic analysis of structural systems with support for multiple analysis methods and advanced model interpretation.

## Features

- **Multi-case Analysis**: Supports Response Spectrum (RS) and Time History (TH) analysis methods
- **Advanced ML Algorithms**: 
  - Tree-based models (Random Forest, XGBoost, LightGBM, CatBoost)
  - Neural Networks (Feedforward with mixed-precision training)
  - Ensemble methods (AdaBoost, Bagging Regressor)
  - Support Vector Regression
  - Linear Regression
- **Model Optimization**:
  - Bayesian hyperparameter tuning
  - Cross-validation with reproducibility controls
  - Early stopping and regularization
- **Model Interpretation**:
  - SHAP value analysis for feature importance
  - Multi-output regression support
  - Interactive visualization generation
- **Production-ready**:
  - Model persistence (HDF5 & pickle formats)
  - Comprehensive metrics tracking
  - GPU acceleration support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/structural-ml-pipeline.git
cd structural-ml-pipeline
```
2. Install dependencies

```bash
pip install -r requirements.txt
```
### Requirements

- Python 3.8+
- numpy
- pandas
- scikit-learn
- xgboost
- lightgbm
- catboost
- tensorflow
- keras-tuner
- scikit-optimize
- shap
- matplotlib
- joblib
- openpyxl

## Usage

### Data Preparation

Place your dataset in `E:\PhD 2024\ML\dataset new mod2.xlsx` or modify `DATASET_PATH` accordingly.

Ensure proper formatting of seismic zone and soil type mappings.

### Configuration

Modify global parameters in the script:

```python
LABEL = "experiment-name"  # Unique identifier for runs
OUTPUT_FEATURES = ['Vb', 'Mb', 'dmax', 'T']  # Target variables
TRAINING_ROWS_COUNT = 500  # Set to 0 for full dataset
```
### Execution

Run the pipeline:

```bash
python structural_ml_pipeline.py
```
### Output Files

- **Best Parameters**: `best parameters {LABEL}.csv`
- **Performance Metrics**: `model metrics {LABEL}.csv`
- **Saved Models**:
  - Neural Networks: `FNN_{CASE} {LABEL}.h5`
  - Other Models: `{MODEL}_{CASE} {LABEL}.pkl`
- **SHAP Visualizations**: `SHAP_{CASE}_{MODEL}_{LABEL}.png`

## Key Configuration Options

| Parameter                | Description                                     | Default Value          |
|--------------------------|-------------------------------------------------|------------------------|
| `DATASET_PATH`           | Path to structural analysis dataset             | `E:\PhD 2024\ML\...`    |
| `TIME_HISTORY_PROPERTIES`| Earthquake ground motion characteristics         | 10 records defined     |
| `ZONE_MAPPING`           | Seismic zone encoding                           | II-IV mapped to 1-4    |
| `SOIL_MAPPING`           | Soil type encoding                              | A-C mapped to 1-3      |
| `TRAINING_ROWS_COUNT`    | Subset size for development (0 = full)          | 500                    |

## License

Proprietary License - Copyright (c) 2023 A.Ayoubi. All rights reserved.

## Contact

**Author:** Abdul Aziz Alayoubi  
**Email:** aziz.ayoubi@outlook.com

For technical inquiries or collaboration opportunities, please contact the author via email.
