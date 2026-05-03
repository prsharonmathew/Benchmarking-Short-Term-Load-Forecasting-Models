"""Project configuration constants."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_URL = (
    "https://data.open-power-system-data.org/time_series/2020-10-06/"
    "time_series_60min_singleindex.csv"
)

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_PATH = RAW_DATA_DIR / "time_series_60min_singleindex.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "load_prepared.csv"
RESULTS_PATH = REPORTS_DIR / "model_results.csv"

TIMESTAMP_COLUMN = "utc_timestamp"

LOAD_COLUMN = "DE_load_actual_entsoe_transparency"

#to fix randomness
RANDOM_STATE = 42
#FInal 20% of the data will be used as test set
TEST_SIZE = 0.20
#Maximum allowed share of missing values in a load column
MAX_MISSING_SHARE = 0.80
#Only interpolate gaps of up to 3 hours.
INTERPOLATION_LIMIT_HOURS = 3

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1h_load",
    "lag_24h_load",
    "lag_168h_load",
    "rolling_mean_24h_load",
    "rolling_std_24h_load",
    "rolling_mean_168h_load",
    "rolling_std_168h_load",
]
