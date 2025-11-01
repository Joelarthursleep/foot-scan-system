import math
from datetime import datetime, timedelta

import pytest

from comprehensive_enhanced_analysis import calculate_proper_health_score
from temporal_comparison_enhanced import extrapolate_health_trajectory


def test_calculate_proper_health_score_no_conditions():
    result = calculate_proper_health_score(
        conditions=[],
        measurements={"length_difference": 0.0, "width_difference": 0.0},
        symmetry_score=100.0,
        regional_metrics={"volume_asymmetry_percent": 0.0},
        history_scores=[]
    )
    assert result["overall_score"] == pytest.approx(100.0)
    assert result["health_grade"] == "Excellent"
    assert result["penalty_breakdown"]["total"] == pytest.approx(0.0)


def test_calculate_proper_health_score_with_high_severity_and_asymmetry():
    conditions = [
        {"name": "Plantar Fasciitis", "clinical_significance": "High", "confidence": 0.9},
        {"name": "Hallux Valgus", "clinical_significance": "Moderate", "confidence": 0.8},
    ]
    measurements = {"length_difference": 12.0, "width_difference": 9.0}
    regional_metrics = {"volume_asymmetry_percent": 9.5}

    result = calculate_proper_health_score(
        conditions=conditions,
        measurements=measurements,
        symmetry_score=82.0,
        previous_score=88.0,
        regional_metrics=regional_metrics,
        history_scores=[{"timestamp": datetime.now().isoformat(), "score": 88.0}]
    )

    # Baseline 100 minus penalties
    expected_penalty_high = 18 * 0.9
    expected_penalty_moderate = 12 * 0.8
    measurement_penalty = (12.0 - 10) * 0.25 + (9.0 - 8) * 0.35 + (90 - 82.0) * 0.1
    bilateral_penalty = 5.0 + min(3.0, (9.5 - 8) * 0.8)
    expected_score = 100.0 - (
        expected_penalty_high + expected_penalty_moderate + measurement_penalty + bilateral_penalty
    )

    assert result["overall_score"] == pytest.approx(expected_score, rel=1e-3)
    assert result["health_grade"] in {"Good", "Fair"}
    assert result["score_delta"] == pytest.approx(result["overall_score"] - 88.0)
    assert result["trend_direction"] in {"declining", "improving", "stable"}
    assert math.isclose(result["penalty_breakdown"]["bilateral"], bilateral_penalty, rel_tol=1e-3)
    assert result["history_records"], "History records should be appended for longitudinal tracking."


def test_extrapolate_health_trajectory_returns_all_models():
    scan_dates = [
        datetime(2024, 1, 1),
        datetime(2024, 4, 1),
        datetime(2024, 7, 1),
        datetime(2024, 10, 1),
    ]
    health_scores = [90.0, 85.0, 80.0, 75.0]

    future_dates, predictions, statistics = extrapolate_health_trajectory(scan_dates, health_scores, 12)

    assert len(future_dates) == 12
    assert len(predictions) == 12
    assert "model_predictions" in statistics
    assert statistics["best_model_key"] in statistics["model_predictions"]

    linear_model = statistics["model_predictions"]["linear"]
    assert linear_model["r2"] > 0.95
    assert linear_model["trend_direction"] == "declining"

    # Ensure predictions continue decline trend
    assert predictions[-1] < health_scores[-1]
