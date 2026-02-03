"""Exploratory Data Analysis on the UCI Wine dataset.

Loads the dataset, computes summary statistics, generates distribution
plots, creates a correlation heatmap, checks class balance, and identifies outliers.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR: str = "output/antigravity"
FIGURE_DPI: int = 150
IQR_MULTIPLIER: float = 1.5
TARGET_COLUMN: str = "target"


def _ensure_output_dir(
    output_dir: str,
) -> Path:
    """Create the output directory if it does not exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_dataset() -> tuple[pl.DataFrame, list[str]]:
    """Load the Wine dataset and return a polars DataFrame and class names."""
    wine = load_wine(as_frame=False)
    feature_names = list(wine.feature_names)
    data = wine.data
    target = wine.target

    df = pl.DataFrame({name: data[:, i] for i, name in enumerate(feature_names)})
    df = df.with_columns(pl.Series(TARGET_COLUMN, target))

    class_names = list(wine.target_names)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    return df, class_names


def _compute_summary_statistics(
    df: pl.DataFrame,
) -> dict[str, Any]:
    """Compute summary statistics for all columns."""
    stats: dict[str, Any] = {}
    for col in df.columns:
        col_data = df[col]
        stats[col] = {
            "mean": round(float(col_data.mean()), 4),
            "median": round(float(col_data.median()), 4),
            "std": round(float(col_data.std()), 4),
            "min": round(float(col_data.min()), 4),
            "max": round(float(col_data.max()), 4),
        }

    logger.info(f"Summary statistics:\n{json.dumps(stats, indent=2, default=str)}")
    return stats


def _check_missing_values(
    df: pl.DataFrame,
) -> dict[str, int]:
    """Check for missing values in each column."""
    missing: dict[str, int] = {}
    for col in df.columns:
        null_count = int(df[col].null_count())
        missing[col] = null_count

    total_missing = sum(missing.values())
    logger.info(f"Total missing values: {total_missing}")
    if total_missing > 0:
        logger.warning(f"Missing values found:\n{json.dumps(missing, indent=2, default=str)}")
    else:
        logger.info("No missing values found in the dataset.")

    return missing


def _plot_distributions(
    df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Generate histogram distribution plots for each feature."""
    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]
    n_cols = 3
    n_rows = (len(feature_columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(feature_columns):
        values = df[col].to_list()
        axes[i].hist(values, bins=40, edgecolor="black", alpha=0.7)
        axes[i].set_title(col)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")

    for j in range(len(feature_columns), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    filepath = output_path / "distributions.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Distribution plots saved to {filepath}")


def _plot_correlation_matrix(
    df: pl.DataFrame,
    output_path: Path,
) -> None:
    """Generate a correlation matrix heatmap for features."""
    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]
    corr_data = {}
    for col in feature_columns:
        correlations = []
        for other_col in feature_columns:
            corr_value = df.select(pl.corr(col, other_col)).item()
            correlations.append(round(float(corr_value), 3))
        corr_data[col] = correlations

    corr_df = pl.DataFrame(corr_data)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_df.to_numpy(),
        annot=False,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=feature_columns,
        yticklabels=feature_columns,
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    filepath = output_path / "correlation_matrix.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Correlation matrix saved to {filepath}")


def _plot_class_balance(
    df: pl.DataFrame,
    class_names: list[str],
    output_path: Path,
) -> dict[str, int]:
    """Plot and return class balance counts."""
    counts = df.select(TARGET_COLUMN).to_series().value_counts()
    counts = counts.sort(TARGET_COLUMN)

    labels = [class_names[int(idx)] for idx in counts[TARGET_COLUMN].to_list()]
    values = counts["count"].to_list()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_title("Class Balance")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.tight_layout()
    filepath = output_path / "class_balance.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Class balance plot saved to {filepath}")

    balance = {label: int(value) for label, value in zip(labels, values)}
    logger.info(f"Class balance:\n{json.dumps(balance, indent=2, default=str)}")
    return balance


def _identify_outliers(
    df: pl.DataFrame,
) -> dict[str, int]:
    """Identify outliers using the IQR method."""
    outlier_counts: dict[str, int] = {}
    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]

    for col in feature_columns:
        q1 = float(df[col].quantile(0.25))
        q3 = float(df[col].quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr

        outlier_count = df.filter((pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)).height
        outlier_counts[col] = outlier_count

    logger.info(
        "Outlier counts (IQR method):\n%s",
        json.dumps(outlier_counts, indent=2, default=str),
    )
    return outlier_counts


def _save_json(
    data: dict[str, Any],
    filepath: Path,
) -> None:
    """Save a dictionary to JSON."""
    filepath.write_text(json.dumps(data, indent=2, default=str))
    logger.info(f"Saved JSON to {filepath}")


def run_eda() -> None:
    """Run the full exploratory data analysis pipeline."""
    start_time = time.time()
    logger.info("Starting exploratory data analysis...")

    output_path = _ensure_output_dir(OUTPUT_DIR)
    df, class_names = _load_dataset()

    summary_stats = _compute_summary_statistics(df)
    missing = _check_missing_values(df)
    class_balance = _plot_class_balance(df, class_names, output_path)
    _plot_distributions(df, output_path)
    _plot_correlation_matrix(df, output_path)
    outliers = _identify_outliers(df)

    _save_json(summary_stats, output_path / "summary_stats.json")
    _save_json(missing, output_path / "missing_values.json")
    _save_json(class_balance, output_path / "class_balance.json")
    _save_json(outliers, output_path / "outlier_counts.json")

    elapsed = time.time() - start_time
    logger.info(f"EDA completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    run_eda()
