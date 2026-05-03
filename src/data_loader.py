"""
Download the CSV file form the URL if it does not already exist, return a much 
more cleaner DataFrame with only the timestamp and best load column.

"""

from __future__ import annotations

import logging
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.config import (
    DATA_URL,
    FIGURES_DIR,
    LOAD_COLUMN,
    MAX_MISSING_SHARE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TIMESTAMP_COLUMN
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedDataset:
    """Container for a loaded demand time series."""

    data: pd.DataFrame
    selected_column: str
    dataset_name: str


class DataLoader:
    """Download, cache, and load demand time-series data."""
    def __init__(self) -> None:
        self.data_url = DATA_URL
        self.raw_data_path = RAW_DATA_PATH
        self.load_column = LOAD_COLUMN

    @staticmethod
    def ensure_directories() -> None:
        """Create project output folders if they are missing."""

        for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, FIGURES_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    def _download_file_if_needed(self, dataset_name: str ="OPSD time-series data") -> None:
        """Download file from the URL if it does not already exist."""

        DataLoader.ensure_directories()
        if self.raw_data_path.exists() and self.raw_data_path.stat().st_size > 0:
            logger.info("Using cached %s: %s", dataset_name, self.raw_data_path)
            return

        tmp_path = self.raw_data_path.with_suffix(f"{self.raw_data_path.suffix}.tmp")
        logger.info("Downloading %s from %s", dataset_name, self.data_url)

        try:
            with urllib.request.urlopen(self.data_url, timeout=60) as response:
                with tmp_path.open("wb") as file_obj:
                    shutil.copyfileobj(response, file_obj)
            tmp_path.replace(self.raw_data_path)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Failed to download {dataset_name} from {self.data_url}") from exc

    def load_electricity_data(self) -> LoadedDataset:
        """Take the raw CSV from the raw_data_path, 
        select the best load column (the one with highest non-null values), 
        and return a cleaned DataFrame."""

        self._download_file_if_needed()


        try:
            header = pd.read_csv(self.raw_data_path, nrows=0)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
            raise RuntimeError(f"Could not read cached OPSD file: {self.raw_data_path}") from exc

        if TIMESTAMP_COLUMN in header.columns:
            timestamp_column = TIMESTAMP_COLUMN
        else:           
            raise RuntimeError(f"Expected timestamp column '{TIMESTAMP_COLUMN}' not found.")
        
        #choose the required columns for analysis.
        available_load_columns = [
            column for column in [self.load_column] if column in header.columns
        ]

        if not available_load_columns:
            raise RuntimeError(
                "None of the configured German load columns were found in the OPSD file."
            )

        usecols = [timestamp_column, *available_load_columns]
        logger.info("Reading selected columns from raw CSV: %s", usecols)
        #read only the required columns to save memory
        try:
            raw = pd.read_csv(self.raw_data_path, usecols=usecols, low_memory=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"Could not read selected columns from OPSD file: {self.raw_data_path}"
            ) from exc

        missing_share = raw[self.load_column].isna().mean()

        if missing_share > MAX_MISSING_SHARE:
            logger.warning(
                f"Load column '{self.load_column}' has {missing_share:.1%} missing values, "
                f"which is above the allowed threshold of {MAX_MISSING_SHARE:.1%}."
            )

        data = raw[[timestamp_column, available_load_columns[0]]].rename(
            columns={timestamp_column: "timestamp", available_load_columns[0]: "load_mw"}
        )

        return LoadedDataset(
            data=data,
            selected_column=available_load_columns[0],
            dataset_name="Germany electricity load",
        )
