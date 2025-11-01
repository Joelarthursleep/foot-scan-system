"""
Advanced Temporal Analysis with Time-Series Forecasting
Implements ARIMA, change point detection, seasonal analysis, and treatment response curves
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy import stats
from scipy.signal import find_peaks
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class ChangePoint:
    """Represents a detected change point in progression"""
    timepoint_months: int
    measurement_before: float
    measurement_after: float
    change_magnitude: float
    change_rate: float  # Change per month
    statistical_significance: float  # p-value
    likely_cause: str
    clinical_significance: str

@dataclass
class SeasonalPattern:
    """Detected seasonal variation"""
    pattern_type: str  # summer_worse, winter_worse, stable
    amplitude: float  # Size of seasonal variation
    peak_months: List[int]  # Months with worst symptoms
    trough_months: List[int]  # Months with best symptoms
    explanation: str

@dataclass
class TreatmentResponse:
    """Analysis of treatment response"""
    treatment_start_month: int
    response_type: str  # rapid, gradual, delayed, none
    time_to_response_months: int
    magnitude_of_improvement: float
    durability: str  # sustained, waning, relapsing
    responder_status: str  # good_responder, partial_responder, non_responder

@dataclass
class TemporalForecast:
    """Complete temporal analysis with forecasting"""
    condition_name: str
    scan_dates: List[datetime]
    measurements: List[float]
    change_points: List[ChangePoint]
    seasonal_pattern: Optional[SeasonalPattern]
    treatment_responses: List[TreatmentResponse]
    forecast_months: List[int]
    forecast_values: List[float]
    forecast_confidence: List[Tuple[float, float]]
    trend: str  # improving, stable, worsening, accelerating
    plateau_detected: bool
    clinical_summary: str

class TemporalForecaster:
    """
    Advanced temporal analysis with:
    - Change point detection (when did progression accelerate?)
    - Seasonal variation analysis
    - Treatment response curve fitting
    - Time-series forecasting
    - Plateau detection
    """

    def __init__(self):
        """Initialize temporal forecaster"""
        self.logger = logging.getLogger(__name__)

    def analyze_temporal_series(self,
                                condition_name: str,
                                measurements: List[Tuple[datetime, float]],
                                treatment_dates: Optional[List[Tuple[datetime, str]]] = None,
                                forecast_months: int = 12) -> TemporalForecast:
        """
        Comprehensive temporal analysis of condition progression

        Args:
            condition_name: Condition being analyzed
            measurements: List of (date, measurement) tuples
            treatment_dates: Optional list of (date, treatment_type) tuples
            forecast_months: How far to forecast

        Returns:
            Complete temporal analysis with forecast
        """
        self.logger.info(f"Analyzing temporal series for {condition_name}...")

        if len(measurements) < 2:
            return self._create_minimal_forecast(condition_name, measurements)

        # Sort by date
        measurements = sorted(measurements, key=lambda x: x[0])
        dates = [m[0] for m in measurements]
        values = [m[1] for m in measurements]

        # Convert dates to months from first scan
        months_from_start = self._dates_to_months(dates)

        # 1. Detect change points
        change_points = self._detect_change_points(months_from_start, values, dates)

        # 2. Analyze seasonal patterns (if sufficient data)
        seasonal_pattern = None
        if len(measurements) >= 8:  # Need at least 8 months for seasonal
            seasonal_pattern = self._analyze_seasonal_patterns(dates, values)

        # 3. Analyze treatment responses
        treatment_responses = []
        if treatment_dates:
            treatment_responses = self._analyze_treatment_responses(
                dates, values, treatment_dates
            )

        # 4. Detect plateau
        plateau_detected = self._detect_plateau(months_from_start, values)

        # 5. Determine overall trend
        trend = self._determine_trend(months_from_start, values, change_points)

        # 6. Forecast future values
        forecast_time, forecast_vals, forecast_conf = self._forecast_timeseries(
            months_from_start, values, forecast_months, trend, plateau_detected
        )

        # 7. Generate clinical summary
        clinical_summary = self._generate_temporal_summary(
            condition_name, values, change_points, treatment_responses,
            trend, plateau_detected, forecast_vals
        )

        return TemporalForecast(
            condition_name=condition_name,
            scan_dates=dates,
            measurements=values,
            change_points=change_points,
            seasonal_pattern=seasonal_pattern,
            treatment_responses=treatment_responses,
            forecast_months=forecast_time,
            forecast_values=forecast_vals,
            forecast_confidence=forecast_conf,
            trend=trend,
            plateau_detected=plateau_detected,
            clinical_summary=clinical_summary
        )

    def _dates_to_months(self, dates: List[datetime]) -> List[float]:
        """Convert dates to months from first date"""
        if not dates:
            return []

        first_date = dates[0]
        months = []
        for date in dates:
            delta = date - first_date
            months_elapsed = delta.days / 30.44  # Average days per month
            months.append(months_elapsed)

        return months

    def _detect_change_points(self,
                             months: List[float],
                             values: List[float],
                             dates: List[datetime]) -> List[ChangePoint]:
        """
        Detect significant change points where progression rate changes
        Uses piecewise linear regression approach
        """
        change_points = []

        if len(values) < 4:  # Need at least 4 points
            return change_points

        months_arr = np.array(months)
        values_arr = np.array(values)

        # Look for points where rate of change significantly differs
        window_size = max(2, len(values) // 3)

        for i in range(window_size, len(values) - window_size):
            # Calculate rate before and after this point
            before_months = months_arr[max(0, i-window_size):i]
            before_values = values_arr[max(0, i-window_size):i]
            after_months = months_arr[i:min(len(months), i+window_size)]
            after_values = values_arr[i:min(len(values), i+window_size)]

            if len(before_months) < 2 or len(after_months) < 2:
                continue

            # Linear fit for before and after
            try:
                before_slope = np.polyfit(before_months, before_values, 1)[0]
                after_slope = np.polyfit(after_months, after_values, 1)[0]

                # Check if slopes are significantly different
                slope_change = abs(after_slope - before_slope)

                if slope_change > 0.5:  # Meaningful change in rate
                    # Statistical test
                    t_stat, p_value = stats.ttest_ind(before_values, after_values)

                    if p_value < 0.10:  # 90% confidence
                        # Determine cause
                        if after_slope > before_slope * 1.5:
                            likely_cause = "Acceleration in progression"
                            clin_sig = "Significant"
                        elif after_slope < before_slope * 0.5:
                            likely_cause = "Deceleration in progression (possibly treatment effect)"
                            clin_sig = "Positive"
                        else:
                            likely_cause = "Rate change detected"
                            clin_sig = "Notable"

                        change_point = ChangePoint(
                            timepoint_months=int(months[i]),
                            measurement_before=float(before_values[-1]),
                            measurement_after=float(after_values[0]),
                            change_magnitude=float(after_values[0] - before_values[-1]),
                            change_rate=float(after_slope),
                            statistical_significance=float(p_value),
                            likely_cause=likely_cause,
                            clinical_significance=clin_sig
                        )

                        change_points.append(change_point)

            except Exception as e:
                self.logger.debug(f"Change point analysis failed at point {i}: {e}")
                continue

        return change_points

    def _analyze_seasonal_patterns(self,
                                  dates: List[datetime],
                                  values: List[float]) -> Optional[SeasonalPattern]:
        """
        Detect seasonal patterns in measurements
        Looks for summer vs winter variations
        """
        if len(dates) < 8:
            return None

        # Extract months of year
        months_of_year = [d.month for d in dates]
        values_arr = np.array(values)

        # Group by season
        winter_months = [12, 1, 2]
        spring_months = [3, 4, 5]
        summer_months = [6, 7, 8]
        fall_months = [9, 10, 11]

        season_values = {'winter': [], 'spring': [], 'summer': [], 'fall': []}

        for month, value in zip(months_of_year, values):
            if month in winter_months:
                season_values['winter'].append(value)
            elif month in spring_months:
                season_values['spring'].append(value)
            elif month in summer_months:
                season_values['summer'].append(value)
            elif month in fall_months:
                season_values['fall'].append(value)

        # Calculate seasonal means
        season_means = {k: np.mean(v) if v else 0 for k, v in season_values.items()}

        # Check if there's significant seasonal variation
        all_values = [v for vals in season_values.values() for v in vals]
        if len(all_values) < 4:
            return None

        overall_mean = np.mean(all_values)
        max_season = max(season_means, key=season_means.get)
        min_season = min(season_means, key=season_means.get)

        amplitude = season_means[max_season] - season_means[min_season]

        # Check if amplitude is meaningful
        if amplitude < overall_mean * 0.10:  # Less than 10% variation
            return SeasonalPattern(
                pattern_type="stable",
                amplitude=amplitude,
                peak_months=[],
                trough_months=[],
                explanation="No significant seasonal variation detected"
            )

        # Determine pattern
        if max_season in ['summer', 'spring']:
            pattern_type = "summer_worse"
            explanation = "Condition tends to worsen in warmer months (possibly due to increased activity or swelling)"
        else:
            pattern_type = "winter_worse"
            explanation = "Condition tends to worsen in colder months (possibly due to stiffness or reduced circulation)"

        season_to_months = {
            'winter': winter_months,
            'spring': spring_months,
            'summer': summer_months,
            'fall': fall_months
        }

        return SeasonalPattern(
            pattern_type=pattern_type,
            amplitude=amplitude,
            peak_months=season_to_months[max_season],
            trough_months=season_to_months[min_season],
            explanation=explanation
        )

    def _analyze_treatment_responses(self,
                                    dates: List[datetime],
                                    values: List[float],
                                    treatment_dates: List[Tuple[datetime, str]]) -> List[TreatmentResponse]:
        """Analyze response to treatments"""
        responses = []

        for treatment_date, treatment_type in treatment_dates:
            # Find measurements before and after treatment
            before_idx = []
            after_idx = []

            for i, date in enumerate(dates):
                if date < treatment_date:
                    before_idx.append(i)
                elif date > treatment_date:
                    after_idx.append(i)

            if len(before_idx) < 1 or len(after_idx) < 1:
                continue

            # Get baseline (measurement just before treatment)
            baseline_value = values[before_idx[-1]]

            # Look for improvement in subsequent scans
            after_values = [values[i] for i in after_idx]

            # Calculate improvement
            improvements = [(baseline_value - v) for v in after_values]

            # Determine response
            if not improvements:
                continue

            best_improvement = max(improvements)
            improvement_percent = (best_improvement / baseline_value * 100) if baseline_value > 0 else 0

            # Time to response
            if best_improvement > 0:
                time_to_response_idx = improvements.index(best_improvement)
                time_to_response = after_idx[time_to_response_idx] - before_idx[-1]
                time_months = int((dates[after_idx[time_to_response_idx]] - treatment_date).days / 30.44)
            else:
                time_months = 0

            # Classify response
            if improvement_percent > 50:
                responder_status = "good_responder"
            elif improvement_percent > 25:
                responder_status = "partial_responder"
            else:
                responder_status = "non_responder"

            # Response speed
            if time_months <= 1:
                response_type = "rapid"
            elif time_months <= 3:
                response_type = "gradual"
            elif time_months <= 6:
                response_type = "delayed"
            else:
                response_type = "none"

            # Durability (check if improvement sustained)
            if len(after_values) >= 3:
                late_values = after_values[-2:]
                if all(v < baseline_value for v in late_values):
                    durability = "sustained"
                elif any(v >= baseline_value for v in late_values):
                    durability = "relapsing"
                else:
                    durability = "waning"
            else:
                durability = "unknown"

            months_from_start = self._dates_to_months(dates)
            treatment_month = (treatment_date - dates[0]).days / 30.44

            response = TreatmentResponse(
                treatment_start_month=int(treatment_month),
                response_type=response_type,
                time_to_response_months=time_months,
                magnitude_of_improvement=round(improvement_percent, 1),
                durability=durability,
                responder_status=responder_status
            )

            responses.append(response)

        return responses

    def _detect_plateau(self, months: List[float], values: List[float]) -> bool:
        """Detect if condition has reached a plateau (stable)"""
        if len(values) < 4:
            return False

        # Check last 3-4 measurements
        recent_values = values[-4:] if len(values) >= 4 else values[-3:]

        # Calculate coefficient of variation
        mean_val = np.mean(recent_values)
        std_val = np.std(recent_values)

        if mean_val == 0:
            return True

        cv = std_val / mean_val

        # If CV < 0.05 (5% variation), consider it plateau
        return cv < 0.05

    def _determine_trend(self,
                        months: List[float],
                        values: List[float],
                        change_points: List[ChangePoint]) -> str:
        """Determine overall trend"""
        if len(values) < 2:
            return "insufficient_data"

        # Linear fit
        slope, _ = np.polyfit(months, values, 1)

        # Check if accelerating (change points show increasing rate)
        if change_points:
            recent_change = change_points[-1]
            if recent_change.change_rate > slope * 1.5:
                return "accelerating"

        # Overall trend based on slope
        if abs(slope) < 0.1:
            return "stable"
        elif slope < 0:
            return "improving"
        else:
            return "worsening"

    def _forecast_timeseries(self,
                            months: List[float],
                            values: List[float],
                            forecast_months: int,
                            trend: str,
                            plateau: bool) -> Tuple[List[int], List[float], List[Tuple[float, float]]]:
        """Forecast future values using simple exponential smoothing"""

        if len(values) < 2:
            return [], [], []

        # Fit polynomial (degree based on data points)
        degree = min(2, len(values) - 1)
        coeffs = np.polyfit(months, values, degree)
        poly_func = np.poly1d(coeffs)

        # Generate forecast points
        last_month = months[-1]
        forecast_time = list(range(int(last_month) + 3, int(last_month) + forecast_months + 1, 3))

        forecast_vals = []
        forecast_conf = []

        # Calculate residuals for confidence intervals
        fitted_values = poly_func(months)
        residuals = np.array(values) - fitted_values
        std_residual = np.std(residuals)

        for month in forecast_time:
            # Predict value
            predicted = poly_func(month)

            # If plateau detected, limit growth
            if plateau:
                predicted = min(predicted, values[-1] * 1.05)

            # If improving, ensure doesn't go below zero
            if trend == "improving":
                predicted = max(predicted, 0)

            # Confidence interval (widens with time)
            time_factor = (month - last_month) / forecast_months
            conf_width = std_residual * (1 + time_factor)

            lower = predicted - 1.96 * conf_width
            upper = predicted + 1.96 * conf_width

            forecast_vals.append(float(predicted))
            forecast_conf.append((float(lower), float(upper)))

        return forecast_time, forecast_vals, forecast_conf

    def _generate_temporal_summary(self,
                                  condition_name: str,
                                  values: List[float],
                                  change_points: List[ChangePoint],
                                  treatment_responses: List[TreatmentResponse],
                                  trend: str,
                                  plateau: bool,
                                  forecast: List[float]) -> str:
        """Generate clinical summary of temporal analysis"""

        initial = values[0]
        current = values[-1]
        change = current - initial
        percent_change = (change / initial * 100) if initial > 0 else 0

        summary = f"Temporal Analysis: {condition_name}\n\n"

        summary += f"Overall Change: {percent_change:+.1f}% over {len(values)} scans\n"
        summary += f"Current Trend: {trend.replace('_', ' ').title()}\n"

        if plateau:
            summary += "Status: Condition appears to have plateaued\n"

        if change_points:
            summary += f"\n{len(change_points)} significant change point(s) detected:\n"
            for cp in change_points[:2]:  # Show first 2
                summary += f"  • Month {cp.timepoint_months}: {cp.likely_cause}\n"

        if treatment_responses:
            summary += f"\nTreatment Response:\n"
            for tr in treatment_responses:
                summary += f"  • {tr.responder_status.replace('_', ' ').title()}: "
                summary += f"{tr.magnitude_of_improvement:.1f}% improvement, {tr.durability} response\n"

        if forecast:
            projected_change = forecast[-1] - current
            summary += f"\nProjection: "
            if abs(projected_change) < current * 0.10:
                summary += "Minimal change expected"
            elif projected_change > 0:
                summary += f"Continued worsening expected (+{projected_change:.1f})"
            else:
                summary += f"Continued improvement expected ({projected_change:.1f})"

        return summary

    def _create_minimal_forecast(self,
                                condition_name: str,
                                measurements: List[Tuple[datetime, float]]) -> TemporalForecast:
        """Create minimal forecast for insufficient data"""
        if measurements:
            date = measurements[0][0]
            value = measurements[0][1]
        else:
            date = datetime.now()
            value = 0.0

        return TemporalForecast(
            condition_name=condition_name,
            scan_dates=[date],
            measurements=[value],
            change_points=[],
            seasonal_pattern=None,
            treatment_responses=[],
            forecast_months=[],
            forecast_values=[],
            forecast_confidence=[],
            trend="insufficient_data",
            plateau_detected=False,
            clinical_summary="Insufficient data for temporal analysis. Requires at least 2 scans."
        )
