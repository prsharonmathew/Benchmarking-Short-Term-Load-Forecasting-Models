"""
Model definitions and benchmark training utilities.
What is happening in ridge-regression: Ridge regression is regularised linear regression model.
** \hat{y}_i = \beta_0 + \beta_j x_{i,j} **
From the training data, it learns the std mean required to standardise the data and the coefficent 
for each feature. Then during preduction it uses these learned information to predict new load values.

Four models are implemented:
1. NaiveSeasonalBaseline: This model simply predicts the next hour's load to be the same as the load
observed 24 hours ago.
2. Ridge Regression: This is a linear regression model with L2 regularization, which helps prevent 
overfitting by adding a penalty term to the loss function based on the magnitude of the coefficients
3. Random Forest: This is an ensemble learning method that constructs multiple decision trees during
training and outputs the mean prediction of the individual trees. It can capture non-linear relationships
between features and the target variable.
4. HistGradientBoosting: This is a gradient boosting algorithm that builds an ensemble of decision 
trees sequentially, where each tree tries to correct the errors of the previous ones. It is designed 
to be efficient and can handle large datasets with high cardinality features.
Finally,
train_and_predict_models returns a dictionary of trained models, each containing the model's name, 
the fitted estimator and its predictions on the test set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE


class PredictiveModel(Protocol):
    """Small protocol for sklearn-like estimators used in this benchmark."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PredictiveModel":
        ...
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

class NaiveSeasonalBaseline:
    """Predict next-hour demand with the same target hour from yesterday."""

    def __init__(self, feature_name: str = "lag_24h_load") -> None:
        self.feature_name = feature_name

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NaiveSeasonalBaseline":
        """
        There is nothing to learn in the "NaiveSeasonalBaseline" model. Here prediction for the next
        hour is simply the load observed 24 hours ago. The fit method is implemented to conform to 
        the PredictiveModel protocol, but it does not perform any operations.
        """
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Here feature name is "lag_24h_load", which means the predicted value for the target hour is
        the load for the same hour 24 hoours ago.
        The predicted value is returned as a numpy array.
        """
        if self.feature_name not in X.columns:
            raise ValueError(f"Missing required feature for naive baseline: {self.feature_name}")
        return X[self.feature_name].to_numpy()

@dataclass
class TrainedModel:
    """A trained model with its test-set predictions."""

    name: str
    estimator: PredictiveModel
    predictions: np.ndarray


def build_model_registry(random_state: int = RANDOM_STATE) -> dict[str, PredictiveModel]:
    """Create the set of benchmark models."""

    return {
        "Naive 24h Baseline": NaiveSeasonalBaseline(),
        "Ridge Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0))
            ]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=80,
            max_depth=16,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=random_state
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=random_state
        )
    }

def train_and_predict_models(
    models: dict[str, PredictiveModel],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> dict[str, TrainedModel]:
    """Fit each benchmark model and generate test-set predictions."""

    trained_models: dict[str, TrainedModel] = {}
    for name, estimator in models.items():
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        trained_models[name] = TrainedModel(
            name=name,
            estimator=estimator,
            predictions=np.asarray(predictions),
        )
    return trained_models
