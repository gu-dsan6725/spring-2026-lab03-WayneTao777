"""Feature engineering for the UCI Wine dataset.

Creates derived features, scales numeric columns, and splits
into stratified training and test sets.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import polars as pl
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR: str = "output/antigravity"
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
TARGET_COLUMN: str = "target"
SCALER_FILENAME: str = "standard_scaler.joblib"

DERIVED_FEATURES: dict[str, str] = {
    "AlcoholToAsh": "alcohol / ash",
    "FlavanoidsToPhenols": "flavanoids / total_phenols",
    "ColorIntensityToHue": "color_intensity / hue",
}


def _ensure_output_dir(
    output_dir: str,
) -> Path:
    """Create the output directory if it does not exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_dataset() -> pl.DataFrame:
    """Load the Wine dataset and return as a polars DataFrame."""
    wine = load_wine(as_frame=False)
    feature_names = list(wine.feature_names)
    data = wine.data
    target = wine.target

    df = pl.DataFrame({name: data[:, i] for i, name in enumerate(feature_names)})
    df = df.with_columns(pl.Series(TARGET_COLUMN, target))

    logger.info(f"Loaded dataset with shape: {df.shape}")
    return df


def _create_derived_features(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Create new ratio-based features from existing columns."""
    df = df.with_columns(
        [
            (pl.col("alcohol") / pl.col("ash")).alias("AlcoholToAsh"),
            (pl.col("flavanoids") / pl.col("total_phenols")).alias("FlavanoidsToPhenols"),
            (pl.col("color_intensity") / pl.col("hue")).alias("ColorIntensityToHue"),
        ]
    )

    logger.info(f"Created {len(DERIVED_FEATURES)} derived features. New shape: {df.shape}")
    logger.info(f"Columns: {df.columns}")
    return df


def _handle_non_finite_values(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Replace non-finite values with column medians."""
    for col in df.columns:
        if df[col].dtype in [pl.Float64, pl.Float32]:
            finite_values = df[col].filter(df[col].is_finite())
            median_val = float(finite_values.median())

            df = df.with_columns(
                pl.when(pl.col(col).is_finite()).then(pl.col(col)).otherwise(median_val).alias(col)
            )

    logger.info("Replaced non-finite values with column medians")
    return df


def _fill_nulls_with_median(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Fill nulls in float columns with column medians."""
    for col in df.columns:
        if df[col].dtype in [pl.Float64, pl.Float32]:
            median_val = float(df[col].median())
            df = df.with_columns(pl.col(col).fill_null(median_val).alias(col))

    logger.info("Filled nulls with column medians")
    return df


def _scale_features(
    df: pl.DataFrame,
    target_column: str,
) -> tuple[pl.DataFrame, StandardScaler]:
    """Scale numeric features using StandardScaler."""
    feature_columns = [c for c in df.columns if c != target_column]

    scaler = StandardScaler()
    feature_values = df.select(feature_columns).to_numpy()
    scaled_values = scaler.fit_transform(feature_values)

    scaled_df = pl.DataFrame({col: scaled_values[:, i] for i, col in enumerate(feature_columns)})
    scaled_df = scaled_df.with_columns(df[target_column])

    logger.info(f"Scaled {len(feature_columns)} features using StandardScaler")
    return scaled_df, scaler


def _split_data(
    df: pl.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split data into stratified training and test sets."""
    feature_columns = [c for c in df.columns if c != target_column]

    x_data = df.select(feature_columns).to_numpy()
    y_data = df[target_column].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=test_size,
        random_state=random_state,
        stratify=y_data,
    )

    x_train_df = pl.DataFrame({col: x_train[:, i] for i, col in enumerate(feature_columns)})
    x_test_df = pl.DataFrame({col: x_test[:, i] for i, col in enumerate(feature_columns)})
    y_train_df = pl.DataFrame({target_column: y_train})
    y_test_df = pl.DataFrame({target_column: y_test})

    logger.info(f"Train set: {x_train_df.shape[0]} samples")
    logger.info(f"Test set: {x_test_df.shape[0]} samples")

    return x_train_df, x_test_df, y_train_df, y_test_df


def _save_splits(
    x_train: pl.DataFrame,
    x_test: pl.DataFrame,
    y_train: pl.DataFrame,
    y_test: pl.DataFrame,
    output_path: Path,
) -> None:
    """Save train/test splits to parquet files."""
    x_train.write_parquet(output_path / "x_train.parquet")
    x_test.write_parquet(output_path / "x_test.parquet")
    y_train.write_parquet(output_path / "y_train.parquet")
    y_test.write_parquet(output_path / "y_test.parquet")

    logger.info(f"Saved train/test splits to {output_path}")


def _log_class_balance(
    y_train: pl.DataFrame,
    y_test: pl.DataFrame,
) -> dict[str, Any]:
    """Log class balance for train and test sets."""
    train_counts = y_train[TARGET_COLUMN].value_counts().sort(TARGET_COLUMN)
    test_counts = y_test[TARGET_COLUMN].value_counts().sort(TARGET_COLUMN)

    balance = {
        "train": {
            str(int(row[TARGET_COLUMN])): int(row["count"])
            for row in train_counts.iter_rows(named=True)
        },
        "test": {
            str(int(row[TARGET_COLUMN])): int(row["count"])
            for row in test_counts.iter_rows(named=True)
        },
    }

    logger.info(f"Split class balance:\n{json.dumps(balance, indent=2, default=str)}")
    return balance


def _save_json(
    data: dict[str, Any],
    filepath: Path,
) -> None:
    """Save a dictionary to JSON."""
    filepath.write_text(json.dumps(data, indent=2, default=str))
    logger.info(f"Saved JSON to {filepath}")


def run_feature_engineering() -> None:
    """Run the full feature engineering pipeline."""
    start_time = time.time()
    logger.info("Starting feature engineering...")

    output_path = _ensure_output_dir(OUTPUT_DIR)

    df = _load_dataset()
    df = _create_derived_features(df)
    df = _handle_non_finite_values(df)
    df = _fill_nulls_with_median(df)
    scaled_df, scaler = _scale_features(df, TARGET_COLUMN)

    x_train, x_test, y_train, y_test = _split_data(
        scaled_df,
        TARGET_COLUMN,
        TEST_SIZE,
        RANDOM_STATE,
    )

    _save_splits(x_train, x_test, y_train, y_test, output_path)
    class_balance = _log_class_balance(y_train, y_test)
    _save_json(DERIVED_FEATURES, output_path / "derived_features.json")
    _save_json(class_balance, output_path / "split_class_balance.json")

    joblib.dump(scaler, output_path / SCALER_FILENAME)
    logger.info(f"Saved scaler to {output_path / SCALER_FILENAME}")

    elapsed = time.time() - start_time
    logger.info(f"Feature engineering completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    run_feature_engineering()
