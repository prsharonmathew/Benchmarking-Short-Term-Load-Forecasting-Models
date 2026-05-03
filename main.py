"""Run the end-to-end electricity load forecasting benchmark."""

from __future__ import annotations

import logging
from src.config import (
    FEATURE_COLUMNS,
    FIGURES_DIR,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RESULTS_PATH,
    TEST_SIZE,
)
from src.data_loader import DataLoader, LoadedDataset
from src.evaluation import evaluate_predictions, get_best_model_name, save_results
from src.features import FeatureEngineer
from src.models import build_model_registry, train_and_predict_models
from src.preprocessing import preprocess_load_data
from src.visualization import (
    plot_feature_importance,
    plot_historical_load,
    plot_metric_comparison,
    plot_recent_predictions,
    plot_test_predictions,
    plot_train_test_split,
)


def configure_logging() -> None:
    """Configure concise console logging for the pipeline."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )


def chronological_train_test_split(n_rows: int, test_size: float) -> int:
    """Return the split index for a chronological train/test split.
    split_index: number of rows for the training dataset.
    """

    if n_rows < 2:
        raise ValueError("Need at least two modeling rows for a train/test split.")
    split_index = int(n_rows * (1 - test_size))
    if split_index <= 0 or split_index >= n_rows:
        raise ValueError("Invalid split index. Adjust the dataset size or test_size.")
    return split_index


def load_demand_dataset(loader: DataLoader) -> LoadedDataset:
    """Load the primary electricity data, falling back only if the primary path fails."""

    logger = logging.getLogger(__name__)
    try:
        return loader.load_electricity_data()
    except RuntimeError as primary_error:
        logger.warning("Electricity dataset failed: %s", primary_error)
        logger.warning("Trying UCI Bike Sharing Dataset as a demand-forecasting fallback.")

def main() -> None:
    """Execute the full benchmark workflow."""

    configure_logging()
    logger = logging.getLogger(__name__)
    DataLoader.ensure_directories()

    loader = DataLoader()
    loaded_dataset = load_demand_dataset(loader)

    logger.info(
        "Loaded %s using source column '%s'.",
        loaded_dataset.dataset_name,
        loaded_dataset.selected_column,
    )

    clean_data = preprocess_load_data(loaded_dataset.data)
    clean_data.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info("Saved processed data to %s", PROCESSED_DATA_PATH)

    feature_engineer = FeatureEngineer()
    X, y, modeling_data = feature_engineer.transform(clean_data)
    split_index = chronological_train_test_split(len(modeling_data), TEST_SIZE)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    test_data = modeling_data.iloc[split_index:].reset_index(drop=True)

    logger.info(
        "Prepared %s modeling rows: %s train, %s test.",
        len(modeling_data),
        len(X_train),
        len(X_test),
    )

    models = build_model_registry(random_state=RANDOM_STATE)
    trained_models = train_and_predict_models(models, X_train, y_train, X_test)
    results = evaluate_predictions(trained_models, y_test)
    save_results(results, RESULTS_PATH)

    best_model_name = get_best_model_name(results)
    logger.info("Best model by MAE: %s", best_model_name)
    logger.info("\n%s", results.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))

    plot_historical_load(
        clean_data,
        FIGURES_DIR,
        title=f"Historical Demand: {loaded_dataset.dataset_name}",
    )
    plot_train_test_split(modeling_data, split_index, FIGURES_DIR)
    plot_test_predictions(test_data, trained_models, best_model_name, FIGURES_DIR)
    plot_recent_predictions(test_data, trained_models, best_model_name, FIGURES_DIR, days=14)
    plot_metric_comparison(results, FIGURES_DIR)

    if "Random Forest" in trained_models:
        plot_feature_importance(
            "Random Forest",
            trained_models["Random Forest"].estimator,
            FEATURE_COLUMNS,
            FIGURES_DIR,
        )

    logger.info("Saved figures to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
