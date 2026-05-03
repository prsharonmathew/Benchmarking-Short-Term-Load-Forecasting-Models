"""Preprocessing for hourly demand time series.
Data preprocessing: 
Col - timestamp: converted to datetime
Col - load_mw: converted to numeric

Remove rows without a timestamp, sort the data by time, and make sure each timestamp appears only once.

Make Col - timestap, the index column.and enforce hourly frequency.
How missing loads are handled?
Missing values are interpolated according to time.
Interpolation happens only between known values and at a time only 3 consecutive values can be interpolated.
The missing value is interpolated smoothly.
Once missing values are interpolated, the remaining missing values are completely dropped.
"""

from __future__ import annotations

import logging
import pandas as pd
from src.config import INTERPOLATION_LIMIT_HOURS

logger = logging.getLogger(__name__)

def preprocess_load_data(
    data: pd.DataFrame, 
    interpolation_limit_hours: int = INTERPOLATION_LIMIT_HOURS) -> pd.DataFrame:
    """Clean timestamps, enforce hourly spacing, and handle small missing gaps."""

    required_columns = {"timestamp", "load_mw"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")

    prepared = data.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce", utc=True)
    prepared["load_mw"] = pd.to_numeric(prepared["load_mw"], errors="coerce")

    prepared = (
        prepared.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
    )

    before_rows = len(prepared)
    prepared = prepared.set_index("timestamp").asfreq("h")
    inserted_rows = len(prepared) - before_rows
    if inserted_rows > 0:
        logger.info("Inserted %s missing hourly timestamp rows.", inserted_rows)

    missing_before = int(prepared["load_mw"].isna().sum())
    prepared["load_mw"] = prepared["load_mw"].interpolate(
        method="time",
        limit=interpolation_limit_hours,
        limit_area="inside"
    )
    missing_after_interpolation = int(prepared["load_mw"].isna().sum())
    logger.info(
        "Missing load values before interpolation: %s; after interpolation: %s.",
        missing_before,
        missing_after_interpolation,
    )

    prepared = prepared.dropna(subset=["load_mw"]).reset_index()
    prepared = prepared[["timestamp", "load_mw"]]

    if prepared.empty:
        raise RuntimeError("No usable rows remain after preprocessing.")

    return prepared
