"""
Machine Learning Pipeline for Structural Engineering Predictive Modeling

Copyright (c) 2023 A.Ayoubi. All rights reserved.

Author: Abdul Aziz Alayoubi
Email: aziz.ayoubi@outlook.com

This script implements a comprehensive machine learning pipeline for predicting structural
response parameters using various regression algorithms. Key features include:
- Hyperparameter tuning with Bayesian optimization
- Cross-validation with reproducibility controls
- Multi-output regression support
- SHAP value analysis for model interpretability
- Mixed-precision neural network training
- Comprehensive metrics tracking and model persistence

The pipeline supports two analysis cases (Response Spectrum and Time History) with different input feature sets
and compares multiple machine learning algorithms including tree-based models,
neural networks, and ensemble methods.
"""

import concurrent
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from keras.optimizers import Adam
from keras_tuner.src.backend.io import tf
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from scikeras.wrappers import KerasRegressor
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, HistGradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from skopt import BayesSearchCV
from skopt.callbacks import DeltaYStopper
from skopt.space import Real, Categorical, Integer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from xgboost import XGBRegressor
from concurrent.futures import ThreadPoolExecutor

# ======================================================================================
# GLOBAL CONFIGURATIONS
# ======================================================================================
LABEL = "training 10 test" # Identifier for current experiment run
OUTPUT_FEATURES = ['Vb', 'Mb', 'dmax', 'T'] # Target variables for prediction
DATASET_PATH = 'E:\\PhD 2024\\ML\\dataset new mod2.xlsx'  # Source data location
TRAINING_ROWS_COUNT = 500 # Subset size for development (0 = use full dataset)

# Earthquake time history properties database
TIME_HISTORY_PROPERTIES = {
    'ChiChi': {'PGA': 0.361, 'AI': 0.37522, 'Pt': 0.06, 'duration': 11.78},
    'Friuli': {'PGA': 0.3513, 'AI': 0.78025, 'Pt': 0.26, 'duration': 4.24},
    'Hollister': {'PGA': 0.1948, 'AI': 0.25754, 'Pt': 0.38, 'duration': 16.53},
    'ImperialValley': {'PGA': 0.3152, 'AI': 1.2646, 'Pt': 0.14, 'duration': 8.92},
    'Kobe': {'PGA': 0.3447, 'AI': 1.68744, 'Pt': 0.16, 'duration': 12.86},
    'Kocaeli': {'PGA': 0.349, 'AI': 1.32244, 'Pt': 1.4, 'duration': 15.62},
    'Landers': {'PGA': 0.7803, 'AI': 6.58123, 'Pt': 0.08, 'duration': 13.73},
    'LomaPrieta': {'PGA': 0.3674, 'AI': 1.34797, 'Pt': 0.22, 'duration': 11.37},
    'Northridge': {'PGA': 0.5683, 'AI': 2, 'Pt': 0.26, 'duration': 9.06},
    'Trinidad': {'PGA': 0.1936, 'AI': 0.17048, 'Pt': 0.28, 'duration': 7.8},
}

# Categorical encoding mappings
ZONE_MAPPING = {'II': 1, 'III': 2, 'IV': 3, 'V': 4}
SOIL_MAPPING = {'A': 1, 'B': 2, 'C': 3}

# ======================================================================================
# CORE FUNCTIONS
# ======================================================================================
def initialize_file(file_path: str, header: str, random_row: str) -> None:
    """Initialize output files with experiment metadata and headers.
    Args:
        file_path: Output file path
        header: CSV header row
        random_row: Random seed information for reproducibility
    """
    with open(file_path, 'w') as f:
        f.write(random_row)
        f.write(header)


def calculate_metrics(model, X, y_true, scaler=None):
    """Calculate regression performance metrics with optional inverse scaling.

    Args:
        model: Trained regression model
        X: Input features
        y_true: Ground truth values
        scaler: Optional scaler for inverse transformation

    Returns:
        Tuple of (R² scores, MAE, RMSE) for each output feature
    """
    y_pred = model.predict(X)
    if scaler is not None:
        y_pred = scaler.inverse_transform(y_pred)
        y_true = scaler.inverse_transform(y_true)
    return (
        r2_score(y_true, y_pred, multioutput='raw_values'),
        mean_absolute_error(y_true, y_pred, multioutput='raw_values'),
        np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
    )


def build_fnn(input_dim, units1=64, units2=32, dropout_rate=0.2, l2_rate=0.001):
    """Construct feedforward neural network architecture with regularization.

    Args:
        input_dim: Number of input features
        units1: Neurons in first hidden layer
        units2: Neurons in second hidden layer
        dropout_rate: Dropout probability
        l2_rate: L2 regularization coefficient

    Returns:
        Compiled Keras model with Adam optimizer
    """
    model = Sequential([
        Dense( units1, activation='relu', kernel_regularizer=l2(l2_rate),
              input_shape=(input_dim,)),
        Dropout(dropout_rate),
        Dense(units2, activation='relu', kernel_regularizer=l2(l2_rate)),
        Dropout(dropout_rate),
        Dense(4, activation='linear', kernel_initializer='zeros')
    ])
    optimizer = Adam(clipvalue=2)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def compute_shap_values(i, estimator, model_name, background, test_samples):
    """Parallel SHAP value computation with model-specific explainers.

    Args:
        i: Estimator index for multi-output models
        estimator: Trained model estimator
        model_name: Model type identifier
        background: Reference dataset for SHAP calculations
        test_samples: Evaluation instances to explain

    Returns:
        Tuple of (estimator index, SHAP values array)
    """
    if model_name in ["LGBM", "XGB", "RF", "CatBoost", "DT", "HGB"]:
        print(f"Computing SHAP values for {model_name} - Estimator {i + 1} (TreeExplainer)...")
        explainer = shap.TreeExplainer(estimator, background)
    else:
        print(f"Computing SHAP values for {model_name} - Estimator {i + 1} (KernelExplainer)...")
        explainer = shap.KernelExplainer(estimator.predict, background, n_jobs=-1)
    return i, explainer.shap_values(test_samples)


def plot_shap_summary(shap_values, test_samples, input_features_label, case_name, model_name):
    """Generate multi-panel SHAP summary plot for model interpretation.

    Args:
        shap_values: Computed SHAP values array
        test_samples: Instances used for explanation
        input_features_label: Human-readable feature names
        case_name: Analysis case identifier (RS/TH)
        model_name: Model type identifier
    """
    num_features = len(shap_values)  # Number of model outputs

    # Create figure and subplots
    fig, axes = plt.subplots(nrows=1, ncols=num_features, figsize=(20, 5))

    # Ensure axes is iterable for a single feature case
    if num_features == 1:
        axes = [axes]

    for i in range(num_features):
        plt.sca(axes[i])  # Set the current axis

        # Save existing axes to detect new ones (e.g., colorbar)
        existing_axes = plt.gcf().axes.copy()

        # Plot the SHAP summary plot with sorting disabled
        shap.summary_plot(
            shap_values[i],
            test_samples,
            feature_names=input_features_label,
            sort=False,
            show=False
        )

        # Identify new axes (typically the colorbar) and remove it from all but the last subplot
        new_axes = [ax for ax in plt.gcf().axes if ax not in existing_axes]
        if i != num_features - 1:
            for ax in new_axes:
                ax.remove()

        # Hide y-axis tick labels on all but the first subplot
        if i != 0:
            axes[i].set_yticklabels([])

        # Set title for each subplot (if custom labels provided)
        if OUTPUT_FEATURES:
            axes[i].set_title(f'{OUTPUT_FEATURES[i]}', fontsize=14)

        plt.xlabel("SHAP Value", fontsize=12)
        axes[i].tick_params(axis='both', labelsize=12)

    plt.tight_layout()

    # Save figure
    filename = f"SHAP_{case_name}_{model_name}_{LABEL}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Figure saved as {filename}")

# ======================================================================================
# PIPELINE CONFIGURATION
# ======================================================================================

def create_cases() -> dict:
    """Define feature configurations for different structural analysis cases.

    Returns:
        Dictionary containing:
        - RS: Code-based simplified analysis features
        - TH: Time history analysis features with seismic parameters
    """
    return {
        "RS": {
            "input_features": ['Length', 'HD', 'ColumnStoryHeight', 'ColumnStoryCount',
                               'HwD', 'SeismicZone', 'Soil'],
            "input_features_label": ['L', 'H/L', 'Hs', 'Ns', 'Hw/L', 'Seismic Zone', 'Soil'],
            "output_features": ['VRS', 'MRS', 'URS', 'Ti']
        },
        "TH": {
            "input_features": ['Length', 'HD', 'ColumnStoryHeight', 'ColumnStoryCount',
                               'HwD', 'AI', 'Pt', 'duration'],
            "input_features_label": ['L', 'H/L', 'Hs', 'Ns', 'Hw/L', 'AI', 'Pt', 'SD'],
            "output_features": ['VTH', 'MTH', 'UTH', 'Ti']
        }
    }


def create_models(random_state_regressor: int) -> dict:
    """Configure machine learning models with hyperparameter search spaces.

    Args:
        random_state_regressor: Seed for reproducible model initialization

    Returns:
        Dictionary of model configurations containing:
        - Model instance
        - Hyperparameter search space for Bayesian optimization
        - Special training parameters (for neural networks)
    """
    return {
        "LR": {"model": LinearRegression(), "params": {}},
        "RF": {
            "model": MultiOutputRegressor(RandomForestRegressor(n_jobs=-1, random_state=random_state_regressor)),
            "params": {
                'estimator__n_estimators': Integer(100, 500),
                'estimator__max_depth': Integer(5, 20),
                'estimator__min_samples_split': Integer(2, 5),
                'estimator__min_samples_leaf': Integer(1, 3),
                'estimator__max_features': Categorical(['sqrt', 'log2'])
            }
        },
        "SVR": {
            "model": MultiOutputRegressor(SVR()),
            "params": {
                'estimator__kernel': Categorical(['rbf']),  # 'linear',
                'estimator__C': [1, 10, 100],  # Narrowed upper limit
                'estimator__gamma': Categorical(['scale', 'auto']),
                'estimator__epsilon': Real(0.001, 1, prior='log-uniform'),
            }
        },
        "AdaBoost": {
            "model": MultiOutputRegressor(
                AdaBoostRegressor(estimator=DecisionTreeRegressor(random_state=random_state_regressor),
                                  random_state=random_state_regressor)),
            "params": {
                'estimator__n_estimators': Integer(50, 150),  # Reduced upper limit
                'estimator__learning_rate': Real(0.05, 0.5, prior='log-uniform')  # Intermediate coverage
            }
        },
        "LGBM": {
            "model": MultiOutputRegressor(LGBMRegressor(
                verbose=-1,
                num_threads=-1,
                random_state=random_state_regressor,
                #         device_type='gpu',  # Enable GPU
            )),
            "params": {
                'estimator__n_estimators': Integer(100, 200),
                'estimator__max_depth': Integer(5, 12),
                'estimator__learning_rate': Real(0.05, 0.15, prior='log-uniform'),
                'estimator__num_leaves': Integer(31, 80)  # Focused on practical range
            }
        },
        "HGB": {
            "model": MultiOutputRegressor(HistGradientBoostingRegressor(random_state=random_state_regressor)),
            "params": {
                'estimator__max_iter': Integer(100, 200),
                'estimator__max_depth': Integer(5, 12),
                'estimator__learning_rate': Real(0.05, 0.15, prior='log-uniform')
            }
        },
        "XGB": {
            "model": MultiOutputRegressor(XGBRegressor(
                eval_metric='rmse',
                n_jobs=-1,
                random_state=random_state_regressor,
                #        tree_method='hist',  # Enable GPU
            )),
            "params": {
                'estimator__n_estimators': Integer(100, 200),
                'estimator__max_depth': Integer(5, 12),
                'estimator__learning_rate': Real(0.05, 0.15, prior='log-uniform'),
                'estimator__subsample': Real(0.8, 1.0),
                'estimator__colsample_bytree': Real(0.8, 1.0)
            }
        },
        "CatBoost": {
            "model": MultiOutputRegressor(CatBoostRegressor(
                verbose=0,
                thread_count=-1,
                random_state=random_state_regressor,
                # task_type='GPU',  # Enable GPU
            )),
            "params": {
                'estimator__iterations': Integer(100, 200),
                'estimator__depth': Integer(5, 12),
                'estimator__learning_rate': Real(0.05, 0.15, prior='log-uniform')
            }
        },
        "BR": {
            "model": MultiOutputRegressor(
                BaggingRegressor(verbose=0, estimator=DecisionTreeRegressor(random_state=random_state_regressor),
                                 n_jobs=-1, random_state=random_state_regressor)),
            "params": {
                'estimator__n_estimators': Integer(10, 50),  # Reduced upper limit
                'estimator__max_samples': Real(0.6, 1.0),
                'estimator__max_features': Real(0.6, 1.0)
            }
        },
        "DT": {
            "model": MultiOutputRegressor(DecisionTreeRegressor(random_state=random_state_regressor)),
            "params": {
                'estimator__max_depth': Integer(5, 20),
                'estimator__min_samples_split': Integer(2, 5),
                'estimator__min_samples_leaf': Integer(1, 3)
            }
        },
        "FNN": {
            "model": KerasRegressor(
                model=build_fnn,  # Pass the build function directly
                verbose=0,
                epochs=100,
                batch_size=128
            ),
            "params": {
                'model__units1': Integer(32, 256),
                'model__units2': Integer(16, 128),
                'model__dropout_rate': Real(0.1, 0.5),
                'model__l2_rate': Real(1e-5, 1e-2, prior='log-uniform'),
                'batch_size': Categorical([32, 64, 128]),  # Now in fit params
                'epochs': Integer(50, 150)
            },
            "fit_params": {
                'callbacks': [EarlyStopping(monitor='val_loss', patience=10)]
            }
        }
    }

# ======================================================================================
# MAIN PIPELINE
# ======================================================================================

def run_model_training_pipeline():
    """End-to-end model training and evaluation workflow.

    Key steps:
    1. Configure computational environment
    2. Initialize experiment tracking files
    3. Load and preprocess structural data
    4. Train and tune models for each analysis case
    5. Evaluate performance metrics
    6. Generate model explanations (SHAP values)
    7. Persist models and results
    """
    # Configure TensorFlow for enhanced performance
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    tf.config.optimizer.set_jit(True)

    # Initialize reproducibility controls
    rng = np.random.default_rng()
    random_states = [
        rng.integers(0, 100000) for _ in range(4)
    ]  # split, kfold, regressor, bayes
    # random_states = [1542, 4526, 55236, 45823] # uncomment this line to use static random state

    # File initialization
    file_paths = {
        'params': f"best parameters {LABEL}.csv",
        'metrics': f"model metrics {LABEL}.csv",
    }
    random_row = f"random:, split,{random_states[0]},k fold,{random_states[1]}," \
                 f"regressor,{random_states[2]},bayes,{random_states[3]}\n"

    for file_path, header in zip(
            [file_paths['params'], file_paths['metrics']],
            ["Case,Model,Best Parameters\n",
             "Case,Model,Feature,Train R2,Train MAE,Train RMSE,Test R2,Test MAE,Test RMSE,Training Time (s),Total Time (s)\n"]
    ):
        initialize_file(file_path, header, random_row)

    # Data loading and preprocessing
    data = pd.read_excel(DATASET_PATH)
    if(TRAINING_ROWS_COUNT > 0):
        data = data.head(TRAINING_ROWS_COUNT)

    # Data preprocessing pipeline
    data['HD'] = data['Height'] / data['Length']
    data['HwD'] = data['Height'] * data['WaterFullnessNum'] / data['Length']
    data['SeismicZone'] = data['SeismicZone'].map(ZONE_MAPPING)
    data['Soil'] = data['Soil'].map(SOIL_MAPPING)

    # Merge time history properties
    time_history_df = pd.DataFrame(TIME_HISTORY_PROPERTIES).T.reset_index(names='TimeHistoryFunction')
    data = data.merge(time_history_df, on='TimeHistoryFunction', how='left')

    # Model training loop
    cases = create_cases()
    models = create_models(random_states[2])

    # Model training and evaluation loop
    for case_name, case_details in cases.items():
        # Prepare dataset for current analysis case
        input_features = case_details['input_features']
        output_features = case_details['output_features']

        # Prepare the data for multi-output
        y = data[output_features]
        X = data[input_features]

        for model_name, model_info in models.items():
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[0])
            print(f"Training and tuning {model_name}...")
            model_start_time = time.time()
            isXScaled = False

            if model_name in ["LR", "SVR", "BR", "FNN"]:
                isXScaled = True
                # Scale the data
                scaler_X = MinMaxScaler() if model_name == "FNN" else StandardScaler()
                X_train = scaler_X.fit_transform(X_train)
                X_test = scaler_X.transform(X_test)
                scaler_y = MinMaxScaler() if model_name == "FNN" else StandardScaler()
                y_train = scaler_y.fit_transform(y_train)
                y_test = scaler_y.transform(y_test)

            if model_name == "FNN":
                model_info["model"] = KerasRegressor(build_fn=build_fnn(input_dim=X_train.shape[1]), verbose=0)

            if model_name == "LR":
                # Directly fit and evaluate Linear Regression
                model_info["model"].fit(X_train, y_train)
                best_model = model_info["model"]
            else:
                # Bayesian hyperparameter optimization
                bayes_search = BayesSearchCV(
                    estimator=model_info["model"],
                    search_spaces=model_info["params"],
                    cv=KFold(n_splits=5, shuffle=True, random_state=random_states[1]),
                    scoring='r2', # Optimize for coefficient of determination
                    n_jobs=-1 if model_name != "FNN" else 1,
                    verbose=0,
                    n_iter=50, # Bayesian optimization iterations
                    random_state=random_states[2]
                )
                if model_name == "FNN":
                    bayes_search.fit(X_train, y_train, **model_info["fit_params"])
                else:
                    delta_stopper = DeltaYStopper(delta=0.001, n_best=5)
                    bayes_search.fit(X_train, y_train, callback=[delta_stopper])
                # Get the best model
                best_model = bayes_search.best_estimator_

            # Measure training time for the best model only
            start_time = time.time()
            best_model.fit(X_train, y_train)  # Train the best model on the full training set
            training_time = time.time() - start_time
            model_time = time.time() - model_start_time

            # Calculate metrics for each output feature
            scaler = scaler_y if isXScaled else None
            train_r2, train_mae, train_rmse = calculate_metrics(best_model, X_train, y_train, scaler)
            test_r2, test_mae, test_rmse = calculate_metrics(best_model, X_test, y_test, scaler)

            best_parameters = bayes_search.best_params_ if model_name != "LR" else "NA"
            # Print and write results
            print(f"{model_name} - Best Parameters: {best_parameters}")
            print(f"{model_name} - Train R2: {train_r2}, MAE: {train_mae}, RMSE: {train_rmse}")
            print(f"{model_name} - Test R2: {test_r2}, MAE: {test_mae}, RMSE: {test_rmse}")
            print(f"{model_name} - Training Time: {training_time:.3f} seconds")
            print(f"{model_name} - Model Time: {model_time:.3f} seconds")
            print("-" * 50)

            # Write best parameters to the parameters file
            with open(file_paths['params'], "a") as param_file:
                param_file.write(f"{case_name},{model_name},{best_parameters}\n")

            # Write metrics to the metrics file
            with open(file_paths['metrics'], "a") as metric_file:
                for i in range(len(train_r2)):
                    metric_file.write(
                        f"{case_name},{model_name}, {output_features[i]},{train_r2[i]},{train_mae[i]},{train_rmse[i]},"
                        f"{test_r2[i]},{test_mae[i]},{test_rmse[i]},{training_time:.3f},{model_time:.3f}\n"
                    )

            # Model persistence
            if model_name == "FNN":
                best_model.model_.save(f"{model_name}_{case_name} {LABEL}.h5")  # Keras format
            else:
                joblib.dump(best_model, f"{model_name}_{case_name} {LABEL}.pkl")  # Pickle format

            # SHAP explainability analysis
            if model_name != "LR":
                shap_size = int(TRAINING_ROWS_COUNT / 2) if TRAINING_ROWS_COUNT > 0 else 1000
                if model_name == "FNN":
                    model = best_model.model_
                    # Sample background data (SHAP needs a reference dataset)
                    background = X_train[np.random.choice(X_train.shape[0], shap_size * 2, replace=False)]

                    # Create a SHAP DeepExplainer
                    explainer = shap.DeepExplainer(model, background)

                    # Calculate SHAP values for a sample of test data
                    test_samples = X_test[:shap_size]  # Use first 500 samples for efficiency
                    shap_values = explainer.shap_values(test_samples, check_additivity=False)
                    # Generate interpretation visualizations
                    plot_shap_summary(shap_values, test_samples, case_details['input_features_label'], case_name,
                                      model_name)

                else:
                    model = best_model

                    # with ThreadPoolEx ecutor(max_workers=8) as executor:
                    shap_values = [None] * len(output_features)

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        # Submit tasks to the executor
                        if not isXScaled:
                            scaler_X = StandardScaler()
                            X_train = scaler_X.fit_transform(X_train)
                            X_test = scaler_X.transform(X_test)
                        futures = []
                        for i, estimator in enumerate(model.estimators_):
                            background_nsample = 200 if (model_name == "SVR" and shap_size < 200) else shap_size
                            test_nsample = 200 if (model_name == "SVR" and shap_size < 200) else shap_size
                            background = X_train[:background_nsample]
                            test_samples = X_test[:test_nsample]  # Use first 500 samples for efficiency
                            future = executor.submit(compute_shap_values, i, estimator, model_name, background,
                                                     test_samples)
                            futures.append(future)
                        # Collect results as they complete
                        for future in concurrent.futures.as_completed(futures):
                            i, shap_value = future.result()
                            shap_values[i] = shap_value
                    # Generate interpretation visualizations
                    plot_shap_summary(shap_values, test_samples, case_details['input_features_label'], case_name, model_name)

# ======================================================================================
# EXECUTION CONTROL
# ======================================================================================
if __name__ == '__main__':
    # Suppress non-critical warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    # Execute main pipeline
    run_model_training_pipeline()
    print("All results and models saved.")
