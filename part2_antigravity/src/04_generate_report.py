"""Generate a comprehensive markdown report for the Wine classifier."""

import json
import logging
import time
from pathlib import Path
from typing import Any

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR: str = "output/antigravity"
REPORT_FILENAME: str = "full_report.md"


def _load_json(
    filepath: Path,
) -> dict[str, Any]:
    """Load a JSON file if it exists, otherwise return empty dict."""
    if filepath.exists():
        return json.loads(filepath.read_text())
    logger.warning(f"Missing file: {filepath}")
    return {}


def _format_metric_block(
    metrics: dict[str, Any],
) -> str:
    """Format key metrics into markdown."""
    if not metrics:
        return "No metrics available."

    lines = [
        f"- Accuracy: {metrics.get('accuracy', 'n/a')}",
        f"- Precision (macro): {metrics.get('precision_macro', 'n/a')}",
        f"- Recall (macro): {metrics.get('recall_macro', 'n/a')}",
        f"- F1 (macro): {metrics.get('f1_macro', 'n/a')}",
    ]
    return "\n".join(lines)


def _format_cv_block(
    cv_results: dict[str, Any],
) -> str:
    """Format cross-validation results into markdown."""
    if not cv_results:
        return "No cross-validation results available."

    lines = []
    for metric_name, payload in cv_results.items():
        lines.append(
            f"- {metric_name}: mean={payload.get('mean', 'n/a')}, std={payload.get('std', 'n/a')}"
        )
    return "\n".join(lines)


def _generate_report(
    summary_stats: dict[str, Any],
    class_balance: dict[str, Any],
    derived_features: dict[str, Any],
    split_balance: dict[str, Any],
    cv_results: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    """Generate the report content as markdown."""
    report_lines = [
        "# Wine Classification Report",
        "",
        "## Overview",
        "This report summarizes exploratory data analysis, feature engineering, model training,",
        "and evaluation for the UCI Wine classification task (3 classes).",
        "",
        "## Dataset Summary",
        f"- Total columns in raw dataset (including target): {len(summary_stats)}",
        f"- Class balance: {json.dumps(class_balance, indent=2, default=str)}",
        "",
        "## Feature Engineering",
        "Derived features:",
        f"{json.dumps(derived_features, indent=2, default=str)}",
        "",
        "Train/test split class balance:",
        f"{json.dumps(split_balance, indent=2, default=str)}",
        "",
        "## Cross-Validation Results",
        _format_cv_block(cv_results),
        "",
        "## Test Set Performance",
        _format_metric_block(metrics),
        "",
        "## Artifacts",
        "- Plots: `distributions.png`, `correlation_matrix.png`, `class_balance.png`",
        "- Confusion matrix: `confusion_matrix.png`",
        "- Feature importance: `feature_importance.png`",
        "- Metrics: `metrics.json` and `cv_results.json`",
        "",
        "## Recommendations",
        "- Consider hyperparameter tuning (e.g., randomized search) for higher F1 scores.",
        "- Review feature importance to identify top predictors and potential simplifications.",
        "- Investigate any class imbalance impact if present in deployment data.",
    ]

    return "\n".join(report_lines)


def run_report() -> None:
    """Generate and save the markdown report."""
    start_time = time.time()
    logger.info("Generating report...")

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_stats = _load_json(output_path / "summary_stats.json")
    class_balance = _load_json(output_path / "class_balance.json")
    derived_features = _load_json(output_path / "derived_features.json")
    split_balance = _load_json(output_path / "split_class_balance.json")
    cv_results = _load_json(output_path / "cv_results.json")
    metrics = _load_json(output_path / "metrics.json")

    report = _generate_report(
        summary_stats,
        class_balance,
        derived_features,
        split_balance,
        cv_results,
        metrics,
    )

    report_path = output_path / REPORT_FILENAME
    report_path.write_text(report)
    logger.info(f"Report saved to {report_path}")

    elapsed = time.time() - start_time
    logger.info(f"Report generation completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    run_report()
