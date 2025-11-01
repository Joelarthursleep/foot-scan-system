"""
Early warning analytics for foot health monitoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import pandas as pd
import numpy as np


@dataclass
class EarlyWarningSignal:
    metric: str
    z_score: float
    percentile: float
    trend: str
    recommendation: str


class EarlyWarningAnalytics:
    """Compute z-scores and percentile drift for patients."""

    def __init__(self, clinical_df: pd.DataFrame):
        self.df = clinical_df.copy()
        if not self.df.empty:
            # Handle timestamps with microseconds by using format='ISO8601'
            self.df["timestamp_dt"] = pd.to_datetime(self.df["timestamp"], format='ISO8601', errors='coerce')

    def compute_patient_signals(self, patient_id: str) -> List[EarlyWarningSignal]:
        if self.df.empty:
            return []

        patient_df = self.df[self.df["patient_id"] == patient_id].sort_values("timestamp_dt")
        if patient_df.empty or len(patient_df) < 2:
            return []

        signals: List[EarlyWarningSignal] = []
        metrics = {
            "avg_length": "Average Foot Length (mm)",
            "avg_width": "Average Foot Width (mm)",
            "length_diff": "Length Asymmetry (mm)",
            "width_diff": "Width Asymmetry (mm)"
        }

        cohort_df = self.df.copy()

        for key, label in metrics.items():
            series_patient = patient_df[key].dropna()
            if series_patient.empty:
                continue

            latest_value = series_patient.iloc[-1]

            series_cohort = cohort_df[key].dropna()
            if len(series_cohort) < 5:
                continue

            mean = series_cohort.mean()
            std = series_cohort.std(ddof=1)
            if std == 0:
                continue

            z_score = (latest_value - mean) / std
            percentile = (series_cohort < latest_value).mean() * 100

            trend = "stable"
            if len(series_patient) >= 3:
                slope = np.polyfit(np.arange(len(series_patient)), series_patient, 1)[0]
                if slope > 0.2:
                    trend = "increasing"
                elif slope < -0.2:
                    trend = "decreasing"

            recommendation = "Monitor routinely."
            if abs(z_score) >= 2.0:
                recommendation = "Escalate for clinical review."
            elif abs(z_score) >= 1.0:
                recommendation = "Increase monitoring frequency."

            signals.append(EarlyWarningSignal(
                metric=label,
                z_score=z_score,
                percentile=percentile,
                trend=trend,
                recommendation=recommendation
            ))

        return signals
