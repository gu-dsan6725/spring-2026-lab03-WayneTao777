"""Train and evaluate an XGBoost regression model on California Housing data.

Loads the prepared train/test splits, trains an XGBoost regressor,
evaluates performance, and saves the model and evaluation artifacts.
"""

import json
import logging
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR: str = "output"
MODEL_FILENAME: str = "xgboost_model.joblib"
FIGURE_DPI: int = 150
N_ESTIMATORS: int = 200
MAX_DEPTH: int = 6
LEARNING_RATE: float = 0.1
RANDOM_STATE: int = 42


def _load_splits(
    output_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train/test splits from parquet files."""
    path = Path(output_dir)

    x_train = pl.read_parquet(path / "x_train.parquet").to_numpy()
    x_test = pl.read_parquet(path / "x_test.parquet").to_numpy()
    y_train = pl.read_parquet(path / "y_train.parquet").to_numpy().ravel()
    y_test = pl.read_parquet(path / "y_test.parquet").to_numpy().ravel()

    logger.info(f"Loaded splits: train={x_train.shape}, test={x_test.shape}")
    return x_train, x_test, y_train, y_test


def _train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> XGBRegressor:
    """Train an XGBoost regressor."""
    model = XGBRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_STATE,
    )

    model.fit(x_train, y_train)
    logger.info(
        f"Trained XGBoost model with n_estimators={N_ESTIMATORS}, "
        f"max_depth={MAX_DEPTH}, learning_rate={LEARNING_RATE}"
    )
    return model


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Compute regression evaluation metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # Compute MAPE, avoiding division by zero
    nonzero_mask = y_true != 0
    mape = float(
        np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    )

    metrics = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "mape_percent": round(mape, 2),
    }

    logger.info(f"Evaluation metrics:\n{json.dumps(metrics, indent=2, default=str)}")
    return metrics


def _plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Generate residual analysis plots."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Predicted vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=10)
    axes[0].plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--",
        linewidth=2,
    )
    axes[0].set_xlabel("Actual Values")
    axes[0].set_ylabel("Predicted Values")
    axes[0].set_title("Predicted vs Actual")

    # Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[1].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Predicted Values")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals vs Predicted")

    plt.tight_layout()
    filepath = output_path / "residual_plots.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Residual plots saved to {filepath}")


def _plot_feature_importance(
    model: XGBRegressor,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Generate a feature importance bar chart."""
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        range(len(importances)),
        importances[sorted_indices],
        align="center",
        alpha=0.8,
    )
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(
        [feature_names[i] for i in sorted_indices],
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
    ax.set_title("XGBoost Feature Importance")

    plt.tight_layout()
    filepath = output_path / "feature_importance.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Feature importance plot saved to {filepath}")


def _save_model(
    model: XGBRegressor,
    output_path: Path,
) -> None:
    """Save the trained model to disk."""
    filepath = output_path / MODEL_FILENAME
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")


def _write_evaluation_report(
    metrics: dict,
    output_path: Path,
) -> None:
    """Write an evaluation report to a markdown file."""
    report = "# Model Evaluation Report\n\n"
    report += "## Metrics Summary\n\n"
    report += "| Metric | Value |\n"
    report += "|--------|-------|\n"
    report += f"| RMSE | {metrics['rmse']} |\n"
    report += f"| MAE | {metrics['mae']} |\n"
    report += f"| R-squared | {metrics['r2']} |\n"
    report += f"| MAPE | {metrics['mape_percent']}% |\n\n"
    report += "## Artifacts\n\n"
    report += "- `residual_plots.png`: Predicted vs actual and residual analysis\n"
    report += "- `feature_importance.png`: XGBoost feature importance ranking\n"
    report += "- `xgboost_model.joblib`: Trained model file\n"

    filepath = output_path / "evaluation_report.md"
    filepath.write_text(report)
    logger.info(f"Evaluation report saved to {filepath}")


def run_training_and_evaluation() -> None:
    """Run the full model training and evaluation pipeline."""
    start_time = time.time()
    logger.info("Starting model training and evaluation...")

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = _load_splits(OUTPUT_DIR)

    model = _train_model(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = _compute_metrics(y_test, y_pred)
    _plot_residuals(y_test, y_pred, output_path)

    # Get feature names from the parquet file columns
    feature_names = pl.read_parquet(output_path / "x_train.parquet").columns
    _plot_feature_importance(model, feature_names, output_path)

    _save_model(model, output_path)
    _write_evaluation_report(metrics, output_path)

    elapsed = time.time() - start_time
    logger.info(f"Training and evaluation completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    run_training_and_evaluation()
