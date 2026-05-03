# Benchmarking Machine Learning Models for Short-Term Electricity Load Forecasting

## Motivation

This project is a compact benchmarking prototype for short-term electricity-demand forecasting. It is designed to show how a Data Scientist can move from a raw public time-series dataset to a clean model comparison with reproducible outputs, clear plots, and production-minded project structure.

The goal is to compare models transparently, not to claim a perfect production model.

## Problem Statement

Predict the next hour of electricity load in Germany using historical hourly load data. The benchmark compares simple and machine-learning approaches against a naive seasonal baseline.

The supervised-learning target is:

```text
target_load_mw = load_mw shifted by -1 hour
```

Each model makes a one-step-ahead forecast from features available at the prediction origin.

## Dataset Description

Primary dataset: Open Power System Data time-series dataset.

Preferred file:

```text
time_series_60min_singleindex.csv
```

Source:

```text
https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv
```

The raw CSV is cached locally under `data/raw/` after the first download. The loader reads only the timestamp and candidate German load columns where possible.

If the primary OPSD electricity dataset cannot be downloaded or loaded, the project includes a fallback loader for the UCI Bike Sharing Dataset. That fallback is only used as a technical backup because hourly bike demand is also a time-series demand-forecasting problem.

## Target Variable

The primary target column is:

```text
DE_load_actual_entsoe_transparency
```

Internally it is renamed to:

```text
load_mw
```

The loader selects the first usable German load column from this priority list:

```text
DE_load_actual_entsoe_transparency
DE_LU_load_actual_entsoe_transparency
DE_tennet_load_actual_entsoe_transparency
DE_amprion_load_actual_entsoe_transparency
DE_50hertz_load_actual_entsoe_transparency
DE_transnetbw_load_actual_entsoe_transparency
```

## Feature Engineering

The feature set is intentionally explainable and aligned to the forecasted target hour:

- Calendar features: `hour`, `day_of_week`, `month`, `is_weekend`
- Lag features: `lag_1h_load`, `lag_24h_load`, `lag_168h_load`
- Rolling features: `rolling_mean_24h_load`, `rolling_std_24h_load`, `rolling_mean_168h_load`, `rolling_std_168h_load`

For example, `lag_24h_load` means the load from 24 hours before the forecasted target timestamp. These features capture daily and weekly seasonality while keeping the benchmark easy to explain in a short interview presentation.

## Leakage Prevention and Validation Strategy

I used chronological splitting to avoid training on future information.

The prepared modeling data is split as:

- First 80%: training set
- Final 20%: test set

The time series is never shuffled. Lag and rolling features are created only from observations available at or before the prediction origin. Because the target is next-hour load, the current observed load is a valid `lag_1h_load` feature relative to the target hour. The rolling windows end at the prediction origin and therefore do not include the future target value.

## Models Benchmarked

The benchmark includes:

- Naive 24-hour seasonal baseline
- Ridge Regression with scaling
- Random Forest Regressor
- HistGradientBoostingRegressor

The naive baseline is important because every ML model should beat a simple operational benchmark.

## Evaluation Metrics

The project calculates:

- MAE
- RMSE
- MAPE with safe zero handling
- R2

The best model is selected by lowest MAE because MAE is easy to interpret in the original demand units.

## Results Summary

The benchmark run selected `DE_load_actual_entsoe_transparency` as the Germany load column and prepared 50,232 supervised modeling rows. The best model by MAE was `HistGradientBoosting`.

| Model | MAE | RMSE | MAPE | R2 |
| --- | ---: | ---: | ---: | ---: |
| HistGradientBoosting | 537.05 | 719.03 | 1.03% | 0.995 |
| Random Forest | 559.28 | 765.19 | 1.07% | 0.994 |
| Ridge Regression | 1,282.24 | 1,695.95 | 2.48% | 0.970 |
| Naive 24h Baseline | 4,137.31 | 6,319.26 | 7.83% | 0.588 |

Running `python main.py` recreates:

```text
reports/model_results.csv
```

The results table is sorted by MAE and the best model is logged in the console. The exact ranking can change if the selected load column, dataset version, or fallback dataset changes, but the benchmark is designed to make that comparison transparent and reproducible.

## How to Run the Project

From the project root, create and activate a virtual environment if desired, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the full benchmark:

```bash
python main.py
```

The first run downloads and caches the raw dataset. Later runs reuse the cached file.

## Example Output Files

Metrics:

```text
reports/model_results.csv
```

Figures:

```text
reports/figures/01_historical_load.png
reports/figures/02_train_test_split.png
reports/figures/03_actual_vs_predicted_test_period.png
reports/figures/04_actual_vs_predicted_final_14_days.png
reports/figures/05_model_metric_comparison.png
reports/figures/06_feature_importance_random_forest.png
```

Processed data:

```text
data/processed/load_prepared.csv
```

## How This Matches the Working Student Data Scientist Role

This project connects directly to the role requirements:

- Benchmarking machine-learning models for an internal customer-style forecasting problem
- Building a compact prototype rather than a one-off notebook
- Verifying model performance against a clear naive baseline
- Using clean Python modules, type hints, docstrings, and reusable functions/classes
- Applying time-series validation that respects chronological order
- Creating interview-readable visualizations and a clear results table
- Preparing the structure for future production work, code review, and Git-based collaboration
- Connecting to energy and grid-related demand forecasting, which matches my academic and research background

## Possible Next Steps

- Add TimeSeriesSplit cross-validation for more robust backtesting
- Include weather, holiday, and renewable-generation features
- Add probabilistic forecast intervals
- Add model serialization and a small prediction API
- Track experiments with MLflow or a lightweight experiment log
- Add unit tests for feature leakage and data-loading edge cases
- Compare optional models such as LightGBM, XGBoost, or SARIMAX if runtime allows
