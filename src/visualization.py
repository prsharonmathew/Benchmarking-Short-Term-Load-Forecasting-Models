"""Visualization helpers for the forecasting benchmark."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.models import TrainedModel


def _prepare_figures_dir(figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def plot_historical_load(
    data: pd.DataFrame,
    figures_dir: Path,
    title: str = "Historical Electricity Load",
) -> Path:
    """Plot the historical cleaned demand time series."""

    _prepare_figures_dir(figures_dir)
    output_path = figures_dir / "01_historical_load.png"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data["timestamp"], data["load_mw"], linewidth=0.8, color="#2f6f9f")
    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Load / demand")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_train_test_split(
    modeling_data: pd.DataFrame,
    split_index: int,
    figures_dir: Path,
    title: str = "Chronological Train/Test Split",
) -> Path:
    """Visualize the chronological split used for validation."""

    _prepare_figures_dir(figures_dir)
    output_path = figures_dir / "02_train_test_split.png"

    train = modeling_data.iloc[:split_index]
    test = modeling_data.iloc[split_index:]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(train["timestamp"], train["target_load_mw"], label="Train", color="#33658a")
    ax.plot(test["timestamp"], test["target_load_mw"], label="Test", color="#c64747")
    ax.axvline(test["timestamp"].iloc[0], color="#444444", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Load / demand")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_test_predictions(
    test_data: pd.DataFrame,
    trained_models: dict[str, TrainedModel],
    best_model_name: str,
    figures_dir: Path,
) -> Path:
    """Plot actual values against the best model and the naive baseline."""

    _prepare_figures_dir(figures_dir)
    output_path = figures_dir / "03_actual_vs_predicted_test_period.png"
    baseline_name = "Naive 24h Baseline"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        test_data["timestamp"],
        test_data["target_load_mw"],
        label="Actual",
        color="#222222",
        linewidth=1.2,
    )
    if baseline_name in trained_models and baseline_name != best_model_name:
        ax.plot(
            test_data["timestamp"],
            trained_models[baseline_name].predictions,
            label=baseline_name,
            color="#9a6b2f",
            linewidth=0.9,
            alpha=0.8,
        )
    ax.plot(
        test_data["timestamp"],
        trained_models[best_model_name].predictions,
        label=f"Best: {best_model_name}",
        color="#247b7b",
        linewidth=0.9,
        alpha=0.9,
    )
    ax.set_title("Actual vs Predicted Load on Test Period")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Load / demand")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_recent_predictions(
    test_data: pd.DataFrame,
    trained_models: dict[str, TrainedModel],
    best_model_name: str,
    figures_dir: Path,
    days: int = 14,
) -> Path:
    """Plot actual and predicted values for the final days of the test set."""

    _prepare_figures_dir(figures_dir)
    output_path = figures_dir / f"04_actual_vs_predicted_final_{days}_days.png"
    baseline_name = "Naive 24h Baseline"
    cutoff = test_data["timestamp"].max() - pd.Timedelta(days=days)
    recent_mask = test_data["timestamp"] >= cutoff
    recent = test_data.loc[recent_mask].copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        recent["timestamp"],
        recent["target_load_mw"],
        label="Actual",
        color="#222222",
        linewidth=1.6,
    )

    if baseline_name in trained_models and baseline_name != best_model_name:
        ax.plot(
            recent["timestamp"],
            trained_models[baseline_name].predictions[recent_mask.to_numpy()],
            label=baseline_name,
            color="#9a6b2f",
            linewidth=1.2,
            alpha=0.8,
        )

    ax.plot(
        recent["timestamp"],
        trained_models[best_model_name].predictions[recent_mask.to_numpy()],
        label=f"Best: {best_model_name}",
        color="#247b7b",
        linewidth=1.2,
    )

    ax.set_title(f"Actual vs Predicted Load: Final {days} Days")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Load / demand")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_metric_comparison(results: pd.DataFrame, figures_dir: Path) -> Path:
    """Create a bar chart comparing MAE and RMSE across models."""

    _prepare_figures_dir(figures_dir)
    output_path = figures_dir / "05_model_metric_comparison.png"

    plot_data = results.sort_values("MAE", ascending=True)
    x_positions = range(len(plot_data))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        [position - width / 2 for position in x_positions],
        plot_data["MAE"],
        width=width,
        label="MAE",
        color="#4f7cac",
    )
    ax.bar(
        [position + width / 2 for position in x_positions],
        plot_data["RMSE"],
        width=width,
        label="RMSE",
        color="#c85a54",
    )
    ax.set_title("Model Error Comparison")
    ax.set_ylabel("Error")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(plot_data["model"], rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_feature_importance(
    model_name: str,
    estimator: object,
    feature_names: list[str],
    figures_dir: Path,
    top_n: int = 10,
) -> Path | None:
    """Plot feature importances for tree-based models that expose them."""

    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return None

    _prepare_figures_dir(figures_dir)
    output_path = figures_dir / f"06_feature_importance_{_safe_filename(model_name)}.png"
    importance_data = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(importance_data["feature"], importance_data["importance"], color="#4f7cac")
    ax.set_title(f"Feature Importance: {model_name}")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
