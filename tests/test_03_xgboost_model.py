"""Tests for the XGBoost model training and evaluation pipeline."""

import importlib
import json

import numpy as np
import pytest
from xgboost import XGBRegressor

# Module name starts with a digit, so use importlib
_module = importlib.import_module("part1_claude_code.demo.solved.03_xgboost_model")
_compute_metrics = _module._compute_metrics
_run_cross_validation = _module._run_cross_validation
_run_hyperparameter_tuning = _module._run_hyperparameter_tuning
_save_tuning_results = _module._save_tuning_results
_write_evaluation_report = _module._write_evaluation_report


# Test fixtures
@pytest.fixture
def sample_data():
    """Create small sample data for testing."""
    rng = np.random.RandomState(42)
    x = rng.rand(100, 5)
    y = x[:, 0] * 2 + x[:, 1] * 3 + rng.rand(100) * 0.1
    return x, y


@pytest.fixture
def trained_model(
    sample_data,
):
    """Return a trained XGBRegressor on sample data."""
    x, y = sample_data
    model = XGBRegressor(
        n_estimators=10,
        max_depth=3,
        random_state=42,
    )
    model.fit(x, y)
    return model


@pytest.fixture
def sample_metrics():
    """Return sample metrics dictionary."""
    return {
        "rmse": 0.4552,
        "mae": 0.2973,
        "r2": 0.8419,
        "mape_percent": 17.02,
    }


class TestComputeMetrics:
    """Tests for _compute_metrics function."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = _compute_metrics(y_true, y_pred)

        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r2"] == 1.0
        assert metrics["mape_percent"] == 0.0

    def test_imperfect_predictions(self):
        """Test metrics with imperfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])

        metrics = _compute_metrics(y_true, y_pred)

        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
        assert 0 < metrics["r2"] < 1
        assert metrics["mape_percent"] > 0

    def test_returns_expected_keys(self):
        """Test that all expected metric keys are present."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])

        metrics = _compute_metrics(y_true, y_pred)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "mape_percent" in metrics


class TestRunCrossValidation:
    """Tests for _run_cross_validation function."""

    def test_returns_expected_keys(
        self,
        sample_data,
        trained_model,
    ):
        """Test that CV results contain expected keys."""
        x, y = sample_data
        results = _run_cross_validation(x, y, trained_model)

        assert "cv_mean_rmse" in results
        assert "cv_std_rmse" in results
        assert "cv_scores" in results

    def test_cv_scores_length(
        self,
        sample_data,
        trained_model,
    ):
        """Test that CV produces the correct number of fold scores."""
        x, y = sample_data
        results = _run_cross_validation(x, y, trained_model)

        assert len(results["cv_scores"]) == 5

    def test_cv_mean_is_positive(
        self,
        sample_data,
        trained_model,
    ):
        """Test that CV mean RMSE is positive."""
        x, y = sample_data
        results = _run_cross_validation(x, y, trained_model)

        assert results["cv_mean_rmse"] > 0
        assert results["cv_std_rmse"] >= 0


class TestRunHyperparameterTuning:
    """Tests for _run_hyperparameter_tuning function."""

    def test_returns_model_and_search(
        self,
        sample_data,
        monkeypatch,
    ):
        """Test that tuning returns a model and search object."""
        monkeypatch.setattr(_module, "N_ITER_SEARCH", 2)
        monkeypatch.setattr(_module, "CV_FOLDS", 2)

        x, y = sample_data
        model, search = _run_hyperparameter_tuning(x, y)

        assert isinstance(model, XGBRegressor)
        assert hasattr(search, "best_params_")
        assert hasattr(search, "best_score_")

    def test_best_params_are_valid(
        self,
        sample_data,
        monkeypatch,
    ):
        """Test that best params contain expected hyperparameters."""
        monkeypatch.setattr(_module, "N_ITER_SEARCH", 2)
        monkeypatch.setattr(_module, "CV_FOLDS", 2)

        x, y = sample_data
        _, search = _run_hyperparameter_tuning(x, y)

        assert "n_estimators" in search.best_params_
        assert "max_depth" in search.best_params_
        assert "learning_rate" in search.best_params_


class TestSaveTuningResults:
    """Tests for _save_tuning_results function."""

    def test_saves_json_file(
        self,
        sample_data,
        tmp_path,
        monkeypatch,
    ):
        """Test that tuning results are saved as JSON."""
        monkeypatch.setattr(_module, "N_ITER_SEARCH", 2)
        monkeypatch.setattr(_module, "CV_FOLDS", 2)

        x, y = sample_data
        _, search = _run_hyperparameter_tuning(x, y)

        _save_tuning_results(search, tmp_path)

        filepath = tmp_path / "tuning_results.json"
        assert filepath.exists()

        results = json.loads(filepath.read_text())
        assert "best_params" in results
        assert "best_cv_rmse" in results
        assert "top_10_candidates" in results


class TestWriteEvaluationReport:
    """Tests for _write_evaluation_report function."""

    def test_basic_report(
        self,
        sample_metrics,
        tmp_path,
    ):
        """Test basic report without CV or tuning info."""
        _write_evaluation_report(sample_metrics, tmp_path)

        filepath = tmp_path / "evaluation_report.md"
        assert filepath.exists()

        content = filepath.read_text()
        assert "# Model Evaluation Report" in content
        assert "0.4552" in content
        assert "Cross-Validation" not in content

    def test_report_with_cv_results(
        self,
        sample_metrics,
        tmp_path,
    ):
        """Test report includes CV section when provided."""
        cv_results = {
            "cv_mean_rmse": 0.5,
            "cv_std_rmse": 0.02,
            "cv_scores": [0.48, 0.50, 0.52, 0.49, 0.51],
        }

        _write_evaluation_report(
            sample_metrics,
            tmp_path,
            cv_results=cv_results,
        )

        content = (tmp_path / "evaluation_report.md").read_text()
        assert "Cross-Validation Results" in content
        assert "0.5" in content

    def test_report_with_best_params(
        self,
        sample_metrics,
        tmp_path,
    ):
        """Test report includes tuning section when provided."""
        best_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
        }

        _write_evaluation_report(
            sample_metrics,
            tmp_path,
            best_params=best_params,
        )

        content = (tmp_path / "evaluation_report.md").read_text()
        assert "Best Hyperparameters" in content
        assert "tuning_results.json" in content
        assert "300" in content
