"""Train and evaluate an XGBoost classifier on Wine data.

Loads the prepared train/test splits, trains an XGBoost classifier,
performs cross-validation, evaluates performance, and saves artifacts.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR: str = "output/claude"
MODEL_FILENAME: str = "xgboost_classifier.joblib"
FIGURE_DPI: int = 150
RANDOM_STATE: int = 42

# Model hyperparameters
N_ESTIMATORS: int = 300
MAX_DEPTH: int = 4
LEARNING_RATE: float = 0.08
SUBSAMPLE: float = 0.9
COLSAMPLE_BYTREE: float = 0.9

# Cross-validation constants
CV_FOLDS: int = 5
CV_SCORING: dict[str, str] = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
}


def _load_splits(
    output_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load train/test splits and class labels from parquet files."""
    path = Path(output_dir)

    x_train = pl.read_parquet(path / "x_train.parquet").to_numpy()
    x_test = pl.read_parquet(path / "x_test.parquet").to_numpy()
    y_train = pl.read_parquet(path / "y_train.parquet").to_numpy().ravel()
    y_test = pl.read_parquet(path / "y_test.parquet").to_numpy().ravel()

    class_labels = sorted({int(label) for label in np.unique(y_train)})
    class_names = [f"Class {label}" for label in class_labels]

    logger.info(f"Loaded splits: train={x_train.shape}, test={x_test.shape}")
    return x_train, x_test, y_train, y_test, class_names


def _build_model() -> XGBClassifier:
    """Build the XGBoost classifier with fixed hyperparameters."""
    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        objective="multi:softprob",
        num_class=3,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )

    logger.info(
        "Initialized XGBoost classifier with n_estimators=%s, max_depth=%s, learning_rate=%s",
        N_ESTIMATORS,
        MAX_DEPTH,
        LEARNING_RATE,
    )
    return model


def _run_cross_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model: XGBClassifier,
) -> dict[str, Any]:
    """Run stratified k-fold cross-validation and return metric summaries."""
    logger.info(f"Running {CV_FOLDS}-fold stratified cross-validation...")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = cross_validate(
        model,
        x_train,
        y_train,
        cv=cv,
        scoring=CV_SCORING,
        n_jobs=-1,
        return_train_score=False,
    )

    summary = {}
    for metric_key in CV_SCORING:
        scores = results[f"test_{metric_key}"]
        summary[metric_key] = {
            "mean": round(float(np.mean(scores)), 4),
            "std": round(float(np.std(scores)), 4),
            "scores": [round(float(score), 4) for score in scores],
        }

    logger.info(f"Cross-validation summary:\n{json.dumps(summary, indent=2, default=str)}")
    return summary


def _train_model(
    model: XGBClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> XGBClassifier:
    """Train the model on the full training dataset."""
    model.fit(x_train, y_train)
    logger.info("Model training completed")
    return model


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Compute classification metrics and detailed report."""
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "recall_macro": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics["classification_report"] = report

    logger.info(f"Test metrics:\n{json.dumps(metrics, indent=2, default=str)}")
    return metrics


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Plot and save the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    filepath = output_path / "confusion_matrix.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Confusion matrix saved to {filepath}")


def _plot_feature_importance(
    model: XGBClassifier,
    output_path: Path,
) -> None:
    """Plot and save feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[indices], color="#DD8452")
    ax.set_title("Feature Importances (XGBoost)")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    filepath = output_path / "feature_importance.png"
    plt.savefig(filepath, dpi=FIGURE_DPI)
    plt.close()
    logger.info(f"Feature importance plot saved to {filepath}")


def _save_json(
    data: dict[str, Any],
    filepath: Path,
) -> None:
    """Save a dictionary to JSON."""
    filepath.write_text(json.dumps(data, indent=2, default=str))
    logger.info(f"Saved JSON to {filepath}")


def run_training() -> None:
    """Run the full model training and evaluation pipeline."""
    start_time = time.time()
    logger.info("Starting model training...")

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test, class_names = _load_splits(OUTPUT_DIR)
    model = _build_model()

    cv_results = _run_cross_validation(x_train, y_train, model)
    model = _train_model(model, x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = _compute_metrics(y_test, y_pred, class_names)

    _plot_confusion_matrix(y_test, y_pred, class_names, output_path)
    _plot_feature_importance(model, output_path)

    _save_json(cv_results, output_path / "cv_results.json")
    _save_json(metrics, output_path / "metrics.json")

    joblib.dump(model, output_path / MODEL_FILENAME)
    logger.info(f"Saved model to {output_path / MODEL_FILENAME}")

    elapsed = time.time() - start_time
    logger.info(f"Model training completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    run_training()
