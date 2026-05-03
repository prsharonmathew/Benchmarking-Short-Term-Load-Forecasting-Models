"""Feature engineering for one-step-ahead demand forecasting.
Build the columns (featured columns) that will be used in the future as inputs 
to the model.
"""

from __future__ import annotations

import pandas as pd
from src.config import FEATURE_COLUMNS

class FeatureEngineer:
    """Create leakage-safe calendar, lag, rolling, and target columns."""

    def __init__(self) -> None:
        self.feature_columns = FEATURE_COLUMNS

    def transform(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Build supervised-learning matrices for next-hour forecasting.Each row represents a predicti-
        on made at ``prediction_origin`` for the next hourly timestamp. The target is ``load_mw.shi-
        ft(-1)``. Lag and rolling features use only load observations available at or before the pr-
        ediction origin, never the target hour.
        """

        required_columns = {"timestamp", "load_mw"}
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            raise ValueError(f"Input data is missing required columns: {missing_columns}")

        work = data.sort_values("timestamp").reset_index(drop=True).copy()
        origin_timestamp = pd.to_datetime(work["timestamp"], utc=True)
        target_timestamp = origin_timestamp + pd.Timedelta(hours=1)
        history_at_origin = work["load_mw"]

        features = pd.DataFrame(
            {
                "prediction_origin": origin_timestamp,
                "timestamp": target_timestamp,
                "target_load_mw": history_at_origin.shift(-1),
            }
        )

        features["hour"] = target_timestamp.dt.hour
        features["day_of_week"] = target_timestamp.dt.dayofweek
        features["month"] = target_timestamp.dt.month
        features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)

        features["lag_1h_load"] = self._target_hour_lag(history_at_origin, hours=1)
        features["lag_24h_load"] = self._target_hour_lag(history_at_origin, hours=24)
        features["lag_168h_load"] = self._target_hour_lag(history_at_origin, hours=168)
        features["rolling_mean_24h_load"] = history_at_origin.rolling(
            window=24, min_periods=24
        ).mean()
        features["rolling_std_24h_load"] = history_at_origin.rolling(
            window=24, min_periods=24
        ).std()
        features["rolling_mean_168h_load"] = history_at_origin.rolling(
            window=168, min_periods=168
        ).mean()
        features["rolling_std_168h_load"] = history_at_origin.rolling(
            window=168, min_periods=168
        ).std()

        modeling_data = features.dropna(subset=[*self.feature_columns, "target_load_mw"])
        modeling_data = modeling_data.reset_index(drop=True)

        if modeling_data.empty:
            raise RuntimeError("No modeling rows remain after creating lag features.")

        X = modeling_data[self.feature_columns]
        y = modeling_data["target_load_mw"]
        return X, y, modeling_data

    @staticmethod
    def _target_hour_lag(history_at_origin: pd.Series, hours: int) -> pd.Series:
        """Return load from ``hours`` before the forecasted target timestamp."""

        if hours < 1:
            raise ValueError("Lag hours must be at least 1.")
        return history_at_origin.shift(hours - 1)
