"""
Advanced Progression Prediction Module
Implements ML-based forecasting, non-linear models, and Monte Carlo simulation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProgressionForecast:
    """Forecast for condition progression"""
    condition_name: str
    current_value: float
    time_points: List[int]  # Months from now
    predicted_values: List[float]
    confidence_intervals_lower: List[float]
    confidence_intervals_upper: List[float]
    progression_rate: str  # accelerating, linear, decelerating, stable
    intervention_urgency: str
    clinical_interpretation: str

@dataclass
class MonteCarloSimulation:
    """Monte Carlo simulation result"""
    condition_name: str
    n_simulations: int
    timeline_months: int
    percentile_5: List[float]
    percentile_25: List[float]
    percentile_50: List[float]  # Median
    percentile_75: List[float]
    percentile_95: List[float]
    probability_threshold_exceeded: Dict[str, float]  # Threshold -> probability

@dataclass
class InterventionImpact:
    """Predicted impact of intervention"""
    intervention_type: str
    without_intervention: ProgressionForecast
    with_intervention: ProgressionForecast
    expected_improvement: float  # Percentage improvement
    optimal_timing: int  # Months from now
    cost_benefit_ratio: float

class ProgressionPredictor:
    """
    Advanced progression prediction with:
    - Non-linear trajectory fitting
    - Gaussian Process regression for uncertainty
    - Monte Carlo simulation
    - Intervention impact modeling
    - Acceleration detection
    """

    def __init__(self):
        """Initialize progression predictor"""
        self.logger = logging.getLogger(__name__)

        # Load progression models
        self.progression_models = self._load_progression_models()
        self.intervention_effects = self._load_intervention_effects()

    def _load_progression_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Load evidence-based progression models for different conditions
        Based on longitudinal studies
        """
        return {
            'hallux_valgus': {
                'natural_progression_rate': 2.0,  # degrees per year
                'progression_type': 'accelerating',  # Gets worse faster over time
                'acceleration_factor': 1.15,  # 15% increase per year
                'plateau_threshold': 50.0,  # degrees - progression slows
                'intervention_effect': 0.70  # 70% reduction in progression
            },
            'pes_planus': {
                'natural_progression_rate': 1.5,  # mm arch height loss per year
                'progression_type': 'linear',
                'acceleration_factor': 1.0,
                'plateau_threshold': 5.0,  # mm - bone hits ground
                'intervention_effect': 0.85  # 85% reduction with orthotics
            },
            'plantar_fasciitis': {
                'natural_progression_rate': 0.5,  # severity score increase per year
                'progression_type': 'decelerating',  # Often self-resolving
                'acceleration_factor': 0.90,  # 10% decrease per year
                'plateau_threshold': 5.0,
                'intervention_effect': 0.60  # 60% faster resolution with treatment
            },
            'metatarsalgia': {
                'natural_progression_rate': 1.0,  # pain score per year
                'progression_type': 'linear',
                'acceleration_factor': 1.0,
                'plateau_threshold': 8.0,
                'intervention_effect': 0.75
            },
            'hammertoe': {
                'natural_progression_rate': 1.5,  # degrees per year
                'progression_type': 'accelerating',
                'acceleration_factor': 1.10,
                'plateau_threshold': 40.0,
                'intervention_effect': 0.95  # Surgery highly effective
            }
        }

    def _load_intervention_effects(self) -> Dict[str, Dict[str, float]]:
        """
        Load intervention effectiveness data
        Returns: {condition: {intervention_type: effectiveness_factor}}
        """
        return {
            'hallux_valgus': {
                'conservative': 0.30,  # 30% reduction
                'orthotic': 0.50,
                'surgical': 0.95
            },
            'pes_planus': {
                'orthotic': 0.85,
                'strengthening': 0.40,
                'surgical': 0.80
            },
            'plantar_fasciitis': {
                'stretching': 0.60,
                'orthotic': 0.70,
                'injection': 0.75,
                'surgery': 0.85
            },
            'hammertoe': {
                'padding': 0.20,
                'orthotic': 0.40,
                'surgical': 0.95
            }
        }

    def predict_progression(self,
                          condition_name: str,
                          current_value: float,
                          historical_data: Optional[List[Tuple[int, float]]] = None,
                          forecast_months: int = 36) -> ProgressionForecast:
        """
        Predict future progression of a condition

        Args:
            condition_name: Name of condition
            current_value: Current measurement (angle, height, etc.)
            historical_data: Optional list of (months_ago, value) tuples
            forecast_months: How far to forecast

        Returns:
            Progression forecast with confidence intervals
        """
        self.logger.info(f"Predicting progression for {condition_name}...")

        condition_key = condition_name.lower().replace(' ', '_')
        model_params = self.progression_models.get(condition_key, {
            'natural_progression_rate': 1.0,
            'progression_type': 'linear',
            'acceleration_factor': 1.0,
            'plateau_threshold': 100.0,
            'intervention_effect': 0.5
        })

        # If we have historical data, use it to improve prediction
        if historical_data and len(historical_data) >= 2:
            forecast = self._predict_from_historical(
                condition_name, current_value, historical_data,
                forecast_months, model_params
            )
        else:
            # Use population-based model
            forecast = self._predict_from_model(
                condition_name, current_value, forecast_months, model_params
            )

        return forecast

    def _predict_from_historical(self,
                                condition_name: str,
                                current_value: float,
                                historical_data: List[Tuple[int, float]],
                                forecast_months: int,
                                model_params: Dict[str, Any]) -> ProgressionForecast:
        """Predict using patient's historical data with Gaussian Process"""

        # Prepare data
        months_back = np.array([d[0] for d in historical_data] + [0])  # Add current
        values = np.array([d[1] for d in historical_data] + [current_value])

        # Convert months_back to months_forward (0 is oldest, current is newest)
        timeline_months = months_back.max() - months_back
        X = timeline_months.reshape(-1, 1)
        y = values

        # Fit Gaussian Process
        kernel = ConstantKernel(1.0) * RBF(length_scale=10.0) + WhiteKernel(noise_level=1.0)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10)

        try:
            gp.fit(X, y)

            # Predict future
            future_months = np.arange(0, forecast_months + 1, 3)  # Every 3 months
            future_X = (X.max() + future_months).reshape(-1, 1)

            predictions, std = gp.predict(future_X, return_std=True)

            # Apply plateau constraint
            plateau = model_params['plateau_threshold']
            predictions = np.minimum(predictions, plateau)

            # Confidence intervals (95%)
            lower = predictions - 1.96 * std
            upper = predictions + 1.96 * std

            # Detect progression pattern
            if len(predictions) >= 3:
                rate_of_change = np.diff(predictions)
                if np.all(rate_of_change > 0) and np.all(np.diff(rate_of_change) > 0):
                    pattern = "accelerating"
                elif np.all(rate_of_change > 0) and np.all(np.diff(rate_of_change) < 0):
                    pattern = "decelerating"
                elif np.allclose(rate_of_change, rate_of_change[0], rtol=0.3):
                    pattern = "linear"
                else:
                    pattern = "variable"
            else:
                pattern = model_params['progression_type']

        except Exception as e:
            self.logger.error(f"GP fitting failed: {e}, falling back to model-based prediction")
            return self._predict_from_model(condition_name, current_value, forecast_months, model_params)

        # Determine urgency
        final_value = predictions[-1]
        percent_change = ((final_value - current_value) / current_value) * 100 if current_value > 0 else 0

        if pattern == "accelerating" and percent_change > 50:
            urgency = "HIGH - Accelerating progression detected"
        elif percent_change > 30:
            urgency = "MODERATE - Significant progression expected"
        elif pattern == "stable" or percent_change < 10:
            urgency = "LOW - Minimal progression expected"
        else:
            urgency = "ROUTINE - Monitor as scheduled"

        interpretation = self._generate_interpretation(
            condition_name, current_value, final_value, pattern, forecast_months
        )

        return ProgressionForecast(
            condition_name=condition_name,
            current_value=current_value,
            time_points=future_months.tolist(),
            predicted_values=predictions.tolist(),
            confidence_intervals_lower=lower.tolist(),
            confidence_intervals_upper=upper.tolist(),
            progression_rate=pattern,
            intervention_urgency=urgency,
            clinical_interpretation=interpretation
        )

    def _predict_from_model(self,
                          condition_name: str,
                          current_value: float,
                          forecast_months: int,
                          model_params: Dict[str, Any]) -> ProgressionForecast:
        """Predict using population-based model"""

        progression_type = model_params['progression_type']
        base_rate = model_params['natural_progression_rate']  # per year
        accel_factor = model_params['acceleration_factor']
        plateau = model_params['plateau_threshold']

        # Monthly rate
        monthly_rate = base_rate / 12

        # Generate timeline
        time_points = list(range(0, forecast_months + 1, 3))
        predictions = []

        current = current_value
        for months in time_points:
            if progression_type == 'accelerating':
                # Exponential growth
                factor = accel_factor ** (months / 12)
                predicted = current_value + (monthly_rate * months * factor)
            elif progression_type == 'decelerating':
                # Logarithmic growth (slows down)
                predicted = current_value + (monthly_rate * months / accel_factor ** (months / 12))
            else:  # linear
                predicted = current_value + (monthly_rate * months)

            # Apply plateau
            predicted = min(predicted, plateau)
            predictions.append(predicted)

        # Confidence intervals (wider for longer forecasts)
        uncertainty = [1 + (m / forecast_months) * 0.3 for m in time_points]
        lower = [p * (1 - u * 0.15) for p, u in zip(predictions, uncertainty)]
        upper = [p * (1 + u * 0.15) for p, u in zip(predictions, uncertainty)]

        # Determine urgency
        final_value = predictions[-1]
        percent_change = ((final_value - current_value) / current_value) * 100 if current_value > 0 else 0

        if progression_type == "accelerating" and percent_change > 50:
            urgency = "HIGH - Condition expected to worsen significantly"
        elif percent_change > 30:
            urgency = "MODERATE - Notable progression expected"
        else:
            urgency = "ROUTINE - Standard monitoring appropriate"

        interpretation = self._generate_interpretation(
            condition_name, current_value, final_value, progression_type, forecast_months
        )

        return ProgressionForecast(
            condition_name=condition_name,
            current_value=current_value,
            time_points=time_points,
            predicted_values=predictions,
            confidence_intervals_lower=lower,
            confidence_intervals_upper=upper,
            progression_rate=progression_type,
            intervention_urgency=urgency,
            clinical_interpretation=interpretation
        )

    def run_monte_carlo_simulation(self,
                                  condition_name: str,
                                  current_value: float,
                                  historical_data: Optional[List[Tuple[int, float]]] = None,
                                  n_simulations: int = 1000,
                                  timeline_months: int = 36) -> MonteCarloSimulation:
        """
        Run Monte Carlo simulation for probabilistic forecasting

        Args:
            condition_name: Condition to simulate
            current_value: Current measurement
            historical_data: Historical measurements
            n_simulations: Number of simulation runs
            timeline_months: Forecast horizon

        Returns:
            Monte Carlo simulation with percentile forecasts
        """
        self.logger.info(f"Running Monte Carlo simulation ({n_simulations} runs)...")

        condition_key = condition_name.lower().replace(' ', '_')
        model_params = self.progression_models.get(condition_key, {
            'natural_progression_rate': 1.0,
            'progression_type': 'linear',
            'acceleration_factor': 1.0,
            'plateau_threshold': 100.0
        })

        # Run simulations
        all_trajectories = []
        time_points = list(range(0, timeline_months + 1, 3))

        for _ in range(n_simulations):
            # Add randomness to progression rate
            rate_variance = np.random.normal(1.0, 0.2)  # ±20% variance
            accel_variance = np.random.normal(model_params['acceleration_factor'], 0.1)

            trajectory = []
            current = current_value

            for months in time_points:
                # Random progression
                monthly_rate = (model_params['natural_progression_rate'] / 12) * rate_variance

                if model_params['progression_type'] == 'accelerating':
                    factor = accel_variance ** (months / 12)
                    predicted = current_value + (monthly_rate * months * factor)
                elif model_params['progression_type'] == 'decelerating':
                    predicted = current_value + (monthly_rate * months / accel_variance ** (months / 12))
                else:
                    predicted = current_value + (monthly_rate * months)

                # Add measurement noise
                noise = np.random.normal(0, predicted * 0.05)  # 5% noise
                predicted += noise

                # Apply plateau
                predicted = min(max(predicted, current_value * 0.8), model_params['plateau_threshold'])

                trajectory.append(predicted)

            all_trajectories.append(trajectory)

        # Calculate percentiles
        trajectories_array = np.array(all_trajectories)
        percentile_5 = np.percentile(trajectories_array, 5, axis=0).tolist()
        percentile_25 = np.percentile(trajectories_array, 25, axis=0).tolist()
        percentile_50 = np.percentile(trajectories_array, 50, axis=0).tolist()
        percentile_75 = np.percentile(trajectories_array, 75, axis=0).tolist()
        percentile_95 = np.percentile(trajectories_array, 95, axis=0).tolist()

        # Calculate threshold exceedance probabilities
        thresholds = {
            'mild_threshold': current_value * 1.2,
            'moderate_threshold': current_value * 1.5,
            'severe_threshold': current_value * 2.0
        }

        threshold_probs = {}
        final_values = trajectories_array[:, -1]
        for thresh_name, thresh_val in thresholds.items():
            prob = (final_values > thresh_val).sum() / n_simulations
            threshold_probs[thresh_name] = round(prob, 3)

        return MonteCarloSimulation(
            condition_name=condition_name,
            n_simulations=n_simulations,
            timeline_months=timeline_months,
            percentile_5=percentile_5,
            percentile_25=percentile_25,
            percentile_50=percentile_50,
            percentile_75=percentile_75,
            percentile_95=percentile_95,
            probability_threshold_exceeded=threshold_probs
        )

    def model_intervention_impact(self,
                                 condition_name: str,
                                 current_value: float,
                                 intervention_type: str,
                                 historical_data: Optional[List[Tuple[int, float]]] = None,
                                 forecast_months: int = 36) -> InterventionImpact:
        """
        Model the impact of intervention vs no intervention

        Args:
            condition_name: Condition name
            current_value: Current measurement
            intervention_type: Type of intervention (e.g., 'orthotic', 'surgical')
            historical_data: Historical data
            forecast_months: Forecast horizon

        Returns:
            Comparison of outcomes with/without intervention
        """
        self.logger.info(f"Modeling {intervention_type} intervention impact...")

        # Predict without intervention
        no_intervention = self.predict_progression(
            condition_name, current_value, historical_data, forecast_months
        )

        # Get intervention effectiveness
        condition_key = condition_name.lower().replace(' ', '_')
        interventions = self.intervention_effects.get(condition_key, {})
        effectiveness = interventions.get(intervention_type, 0.5)  # Default 50%

        # Model with intervention (reduce progression rate)
        model_params = self.progression_models.get(condition_key, {})
        modified_params = model_params.copy()
        modified_params['natural_progression_rate'] *= (1 - effectiveness)
        modified_params['acceleration_factor'] = 1 + (modified_params['acceleration_factor'] - 1) * (1 - effectiveness)

        # Predict with intervention
        with_intervention = self._predict_from_model(
            condition_name, current_value, forecast_months, modified_params
        )

        # Calculate improvement
        no_intervention_final = no_intervention.predicted_values[-1]
        with_intervention_final = with_intervention.predicted_values[-1]

        if no_intervention_final > current_value:
            improvement_percent = ((no_intervention_final - with_intervention_final) /
                                  (no_intervention_final - current_value)) * 100
        else:
            improvement_percent = 0

        # Optimal timing (earlier is generally better)
        optimal_timing = 0  # Immediate intervention usually best

        # Simple cost-benefit (improvement per unit cost - placeholder)
        cost_benefit = improvement_percent / (1.0 if intervention_type != 'surgical' else 10.0)

        return InterventionImpact(
            intervention_type=intervention_type,
            without_intervention=no_intervention,
            with_intervention=with_intervention,
            expected_improvement=round(improvement_percent, 1),
            optimal_timing=optimal_timing,
            cost_benefit_ratio=round(cost_benefit, 2)
        )

    def _generate_interpretation(self,
                                condition_name: str,
                                current: float,
                                future: float,
                                pattern: str,
                                months: int) -> str:
        """Generate clinical interpretation of forecast"""

        change = future - current
        percent = (change / current * 100) if current > 0 else 0

        if percent < 10:
            severity = "minimal"
            action = "Continue routine monitoring"
        elif percent < 30:
            severity = "moderate"
            action = "Consider preventive interventions"
        else:
            severity = "significant"
            action = "Recommend proactive treatment to prevent progression"

        interpretation = (
            f"{condition_name} is predicted to show {severity} progression over the next "
            f"{months} months ({percent:.1f}% change). The progression pattern appears to be "
            f"{pattern}. {action}."
        )

        return interpretation
