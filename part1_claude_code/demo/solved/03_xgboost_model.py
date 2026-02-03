"""Train and evaluate an XGBoost regression model on California Housing data.

Loads the prepared train/test splits, trains an XGBoost regressor,
evaluates performance, and saves the model and evaluation artifacts.
Supports cross-validation and hyperparameter tuning via CLI flags.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
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

# Cross-validation constants
CV_FOLDS: int = 5
CV_SCORING: str = "neg_root_mean_squared_error"
N_ITER_SEARCH: int = 50

# Hyperparameter search space
PARAM_DISTRIBUTIONS: dict = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.2, 0.3],
}


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
    """Train an XGBoost regressor with default hyperparameters."""
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


def _run_cross_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model: XGBRegressor,
) -> dict:
    """Run k-fold cross-validation and return score statistics.

    Args:
        x_train: Training feature matrix.
        y_train: Training target array.
        model: Fitted or unfitted XGBRegressor to evaluate.

    Returns:
        Dictionary with cv_mean, cv_std, and cv_scores (RMSE, positive).
    """
    logger.info(f"Running {CV_FOLDS}-fold cross-validation...")

    scores = cross_val_score(
        model,
        x_train,
        y_train,
        cv=CV_FOLDS,
        scoring=CV_SCORING,
        n_jobs=-1,
    )

    # Convert negative RMSE scores to positive
    rmse_scores = -scores

    cv_results = {
        "cv_mean_rmse": round(float(np.mean(rmse_scores)), 4),
        "cv_std_rmse": round(float(np.std(rmse_scores)), 4),
        "cv_scores": [round(float(s), 4) for s in rmse_scores],
    }

    logger.info(
        f"Cross-validation RMSE: {cv_results['cv_mean_rmse']} (+/- {cv_results['cv_std_rmse']})"
    )
    logger.info(f"Per-fold RMSE scores: {cv_results['cv_scores']}")

    return cv_results


def _run_hyperparameter_tuning(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[XGBRegressor, RandomizedSearchCV]:
    """Run randomized search for hyperparameter tuning.

    Args:
        x_train: Training feature matrix.
        y_train: Training target array.

    Returns:
        Tuple of (best estimator, full RandomizedSearchCV object).
    """
    logger.warning(
        f"Starting hyperparameter tuning with {N_ITER_SEARCH} iterations "
        f"and {CV_FOLDS}-fold CV. This may take a while."
    )

    base_model = XGBRegressor(random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER_SEARCH,
        cv=CV_FOLDS,
        scoring=CV_SCORING,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(x_train, y_train)

    best_rmse = -search.best_score_
    logger.info(f"Best CV RMSE: {best_rmse:.4f}")
    logger.info(f"Best parameters:\n{json.dumps(search.best_params_, indent=2, default=str)}")

    return search.best_estimator_, search


def _save_tuning_results(
    search: RandomizedSearchCV,
    output_path: Path,
) -> None:
    """Save hyperparameter tuning results to a JSON file.

    Args:
        search: Fitted RandomizedSearchCV object.
        output_path: Directory to save the results file.
    """
    cv_results = search.cv_results_

    # Build list of all candidates sorted by rank
    candidates = []
    for i in range(len(cv_results["params"])):
        candidates.append(
            {
                "rank": int(cv_results["rank_test_score"][i]),
                "mean_rmse": round(float(-cv_results["mean_test_score"][i]), 4),
                "std_rmse": round(float(cv_results["std_test_score"][i]), 4),
                "params": {
                    k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                    for k, v in cv_results["params"][i].items()
                },
            }
        )

    candidates.sort(key=lambda x: x["rank"])

    results = {
        "best_params": {
            k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
            for k, v in search.best_params_.items()
        },
        "best_cv_rmse": round(float(-search.best_score_), 4),
        "n_iterations": N_ITER_SEARCH,
        "cv_folds": CV_FOLDS,
        "top_10_candidates": candidates[:10],
    }

    filepath = output_path / "tuning_results.json"
    filepath.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Tuning results saved to {filepath}")


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
    cv_results: Optional[dict] = None,
    best_params: Optional[dict] = None,
) -> None:
    """Write an evaluation report to a markdown file.

    Args:
        metrics: Dictionary of test set evaluation metrics.
        output_path: Directory to write the report.
        cv_results: Optional cross-validation results dictionary.
        best_params: Optional best hyperparameters from tuning.
    """
    report = "# Model Evaluation Report\n\n"
    report += "## Metrics Summary\n\n"
    report += "| Metric | Value |\n"
    report += "|--------|-------|\n"
    report += f"| RMSE | {metrics['rmse']} |\n"
    report += f"| MAE | {metrics['mae']} |\n"
    report += f"| R-squared | {metrics['r2']} |\n"
    report += f"| MAPE | {metrics['mape_percent']}% |\n\n"

    if cv_results is not None:
        report += "## Cross-Validation Results\n\n"
        report += f"- **Folds**: {CV_FOLDS}\n"
        report += f"- **Mean RMSE**: {cv_results['cv_mean_rmse']}\n"
        report += f"- **Std RMSE**: {cv_results['cv_std_rmse']}\n"
        report += f"- **Per-fold RMSE**: {cv_results['cv_scores']}\n\n"

    if best_params is not None:
        report += "## Best Hyperparameters (from tuning)\n\n"
        report += "| Parameter | Value |\n"
        report += "|-----------|-------|\n"
        for param, value in sorted(best_params.items()):
            report += f"| {param} | {value} |\n"
        report += "\n"

    report += "## Artifacts\n\n"
    report += "- `residual_plots.png`: Predicted vs actual and residual analysis\n"
    report += "- `feature_importance.png`: XGBoost feature importance ranking\n"
    report += "- `xgboost_model.joblib`: Trained model file\n"

    if best_params is not None:
        report += "- `tuning_results.json`: Hyperparameter tuning results\n"

    filepath = output_path / "evaluation_report.md"
    filepath.write_text(report)
    logger.info(f"Evaluation report saved to {filepath}")


def run_training_and_evaluation(
    tune: bool = False,
    cv_only: bool = False,
) -> None:
    """Run the full model training and evaluation pipeline.

    Args:
        tune: If True, run hyperparameter tuning before training.
        cv_only: If True, run cross-validation on the default model.
    """
    start_time = time.time()
    logger.info("Starting model training and evaluation...")

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = _load_splits(OUTPUT_DIR)

    cv_results = None
    best_params = None

    if tune:
        # Run hyperparameter tuning
        model, search = _run_hyperparameter_tuning(x_train, y_train)
        best_params = search.best_params_
        _save_tuning_results(search, output_path)

        # Run CV on the best model for reporting
        cv_results = _run_cross_validation(x_train, y_train, model)

    elif cv_only:
        # Train default model and run CV
        model = _train_model(x_train, y_train)
        cv_results = _run_cross_validation(x_train, y_train, model)

    else:
        # Default: train with fixed hyperparameters
        model = _train_model(x_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(x_test)
    metrics = _compute_metrics(y_test, y_pred)

    # Generate plots
    _plot_residuals(y_test, y_pred, output_path)
    feature_names = pl.read_parquet(output_path / "x_train.parquet").columns
    _plot_feature_importance(model, feature_names, output_path)

    # Save model and report
    _save_model(model, output_path)
    _write_evaluation_report(
        metrics,
        output_path,
        cv_results=cv_results,
        best_params=best_params,
    )

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    if minutes > 0:
        logger.info(
            f"Training and evaluation completed in {minutes} minutes and {seconds:.1f} seconds"
        )
    else:
        logger.info(f"Training and evaluation completed in {seconds:.1f} seconds")


def main() -> None:
    """Parse CLI arguments and run the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate an XGBoost model on California Housing data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Default training with fixed hyperparameters
    uv run python -m part1_claude_code.solved.03_xgboost_model

    # Run with cross-validation on default model
    uv run python -m part1_claude_code.solved.03_xgboost_model --cv-only

    # Run with hyperparameter tuning
    uv run python -m part1_claude_code.solved.03_xgboost_model --tune
""",
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Run randomized hyperparameter search before training",
    )
    parser.add_argument(
        "--cv-only",
        action="store_true",
        default=False,
        help="Run cross-validation on the default model (no tuning)",
    )

    args = parser.parse_args()

    if args.tune and args.cv_only:
        parser.error("--tune and --cv-only are mutually exclusive")

    run_training_and_evaluation(
        tune=args.tune,
        cv_only=args.cv_only,
    )


if __name__ == "__main__":
    main()
