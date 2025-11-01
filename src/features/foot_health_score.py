"""
Comprehensive Foot Health Scoring System
Calculates an overall foot health score (0-100) for insurance risk assessment
and temporal health decline tracking
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class FootHealthScore:
    """Overall foot health assessment"""
    overall_score: float  # 0-100, where 100 is perfect health
    health_grade: str  # 'Excellent', 'Good', 'Fair', 'Poor', 'Critical'
    percentile_rank: float  # Compared to normal population (0-100)
    category_scores: Dict[str, float]
    risk_level: str  # 'Low', 'Moderate', 'High', 'Critical'
    health_decline_rate: Optional[float] = None  # Points per year (if temporal data available)
    mobility_impact_score: float = 0  # 0-100, higher = worse mobility impact
    fall_likelihood: float = 0  # 0-100 probability
    insurance_risk_factor: float = 1.0  # Multiplier for insurance risk (1.0 = average)
    key_concerns: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskMatrix:
    """Multi-dimensional risk assessment matrix

    NOTE: Only includes risks measurable from 3D LiDAR scan.
    Circulatory and neurological risks require additional sensors.
    """
    structural_risk: Dict[str, any]  # Measurable from 3D scan
    biomechanical_risk: Dict[str, any]  # Measurable from 3D scan
    mobility_risk: Dict[str, any]  # Inferred from structural/biomechanical
    fall_risk: Dict[str, any]  # Inferred from balance and alignment
    symmetry_risk: Dict[str, any]  # Measurable from comparing left/right
    overall_risk_level: str
    critical_flags: List[str]
    unmeasurable_note: str = "Circulatory, neurological, and skin assessments require camera/IR sensors (planned for future deployment)"


class FootHealthScoreCalculator:
    """Calculates comprehensive foot health scores for insurance assessment"""

    # Normal population baselines (derived from clinical literature)
    # NOTE: Only includes metrics measurable from 3D LiDAR scan
    NORMAL_BASELINES = {
        'structural_integrity': 90.0,
        'biomechanical_function': 85.0,
        'symmetry': 92.0,  # Left vs Right foot symmetry
    }

    # Weights for overall score calculation
    # NOTE: These are the ONLY metrics we can measure from a 3D scan
    CATEGORY_WEIGHTS = {
        'structural_integrity': 0.50,  # Bunions, arch, toe deformities (measurable from scan)
        'biomechanical_function': 0.35,  # Gait, pronation, alignment (measurable from scan)
        'symmetry': 0.15,  # Left/right differences (measurable from scan)
    }

    # Condition severity weights (how much each condition affects the score)
    CONDITION_SEVERITY_IMPACT = {
        'none': 0,
        'mild': 10,
        'moderate': 25,
        'severe': 50,
    }

    def __init__(self):
        """Initialize the calculator"""
        self.logger = logging.getLogger(__name__)

    def calculate_comprehensive_score(
        self,
        medical_conditions: Dict,
        risk_assessments: Optional[Dict] = None,
        patient_data: Optional[Dict] = None,
        point_cloud: Optional[np.ndarray] = None,
        segmentation: Optional[np.ndarray] = None
    ) -> FootHealthScore:
        """Calculate overall foot health score from all available data"""

        # Calculate category scores
        category_scores = self._calculate_category_scores(
            medical_conditions, patient_data, point_cloud, segmentation
        )

        # Calculate weighted overall score
        overall_score = self._calculate_weighted_score(category_scores)

        # Determine health grade
        health_grade = self._get_health_grade(overall_score)

        # Calculate percentile rank
        percentile_rank = self._calculate_percentile_rank(overall_score, category_scores)

        # Determine overall risk level
        risk_level = self._determine_risk_level(overall_score)

        # Calculate mobility impact
        mobility_impact = self._calculate_mobility_impact(medical_conditions, category_scores)

        # Calculate fall likelihood
        fall_likelihood = self._calculate_fall_likelihood(
            medical_conditions, category_scores, risk_assessments
        )

        # Calculate insurance risk factor
        insurance_risk_factor = self._calculate_insurance_risk_factor(
            overall_score, mobility_impact, fall_likelihood, risk_assessments
        )

        # Identify key concerns and strengths
        key_concerns = self._identify_key_concerns(category_scores, medical_conditions)
        strengths = self._identify_strengths(category_scores)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            health_grade, key_concerns, risk_level
        )

        return FootHealthScore(
            overall_score=overall_score,
            health_grade=health_grade,
            percentile_rank=percentile_rank,
            category_scores=category_scores,
            risk_level=risk_level,
            mobility_impact_score=mobility_impact,
            fall_likelihood=fall_likelihood,
            insurance_risk_factor=insurance_risk_factor,
            key_concerns=key_concerns,
            strengths=strengths,
            recommendations=recommendations
        )

    def _calculate_category_scores(
        self,
        medical_conditions: Dict,
        patient_data: Optional[Dict],
        point_cloud: Optional[np.ndarray],
        segmentation: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Calculate scores for each health category

        NOTE: Only calculates metrics that can be measured from 3D LiDAR scan.
        Excludes neurological, circulatory, and skin health as these require
        additional sensors (thermal/IR cameras, sensory testing equipment).
        """

        scores = {}

        # Structural Integrity Score (measurable from 3D scan)
        scores['structural_integrity'] = self._calculate_structural_score(medical_conditions)

        # Biomechanical Function Score (measurable from 3D scan)
        scores['biomechanical_function'] = self._calculate_biomechanical_score(
            medical_conditions, point_cloud, segmentation
        )

        # Symmetry Score (measurable from comparing left/right scans)
        scores['symmetry'] = self._calculate_symmetry_score(medical_conditions, patient_data)

        return scores

    def _calculate_structural_score(self, medical_conditions: Dict) -> float:
        """Calculate structural integrity score"""
        base_score = 100.0

        structural_conditions = [
            'hallux_valgus', 'hallux_rigidus', 'hammer_toe', 'claw_toe',
            'bunion', 'flat_foot', 'cavus_foot', 'metatarsalgia'
        ]

        for cond_name, condition in medical_conditions.items():
            if any(sc in cond_name.lower() for sc in structural_conditions):
                if hasattr(condition, 'detected') and condition.detected:
                    severity = getattr(condition, 'severity', 'mild')
                    impact = self.CONDITION_SEVERITY_IMPACT.get(severity, 10)
                    base_score -= impact

        return max(0, min(100, base_score))

    def _calculate_symmetry_score(
        self,
        medical_conditions: Dict,
        patient_data: Optional[Dict]
    ) -> float:
        """Calculate left/right foot symmetry score (measurable from 3D scan)"""
        base_score = 100.0

        # Check for asymmetry data from patient_data
        if patient_data:
            length_diff = patient_data.get('length_difference', 0)
            width_diff = patient_data.get('width_difference', 0)

            # Penalize significant asymmetries
            if length_diff > 10.0:  # More than 10mm difference
                base_score -= 30
            elif length_diff > 5.0:  # More than 5mm difference
                base_score -= 15

            if width_diff > 5.0:  # More than 5mm difference
                base_score -= 20
            elif width_diff > 3.0:  # More than 3mm difference
                base_score -= 10

        return max(0, min(100, base_score))

    def _calculate_biomechanical_score(
        self,
        medical_conditions: Dict,
        point_cloud: Optional[np.ndarray],
        segmentation: Optional[np.ndarray]
    ) -> float:
        """Calculate biomechanical function score"""
        base_score = 100.0

        biomechanical_conditions = [
            'plantar_fasciitis', 'achilles', 'gait', 'pronation',
            'supination', 'arch', 'heel_spur'
        ]

        for cond_name, condition in medical_conditions.items():
            if any(bc in cond_name.lower() for bc in biomechanical_conditions):
                if hasattr(condition, 'detected') and condition.detected:
                    severity = getattr(condition, 'severity', 'mild')
                    impact = self.CONDITION_SEVERITY_IMPACT.get(severity, 10)
                    base_score -= impact

        return max(0, min(100, base_score))


    def _calculate_weighted_score(self, category_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        total_score = 0
        for category, score in category_scores.items():
            weight = self.CATEGORY_WEIGHTS.get(category, 0)
            total_score += score * weight

        return round(total_score, 1)

    def _get_health_grade(self, overall_score: float) -> str:
        """Determine health grade based on score"""
        if overall_score >= 90:
            return 'Excellent'
        elif overall_score >= 75:
            return 'Good'
        elif overall_score >= 60:
            return 'Fair'
        elif overall_score >= 40:
            return 'Poor'
        else:
            return 'Critical'

    def _calculate_percentile_rank(
        self,
        overall_score: float,
        category_scores: Dict[str, float]
    ) -> float:
        """Calculate percentile rank compared to normal population"""

        # Compare to normal baseline (assume normal distribution)
        normal_mean = 82.0  # Average healthy foot
        normal_std = 12.0   # Standard deviation

        # Calculate z-score
        z_score = (overall_score - normal_mean) / normal_std

        # Convert to percentile (approximate)
        from scipy import stats
        percentile = stats.norm.cdf(z_score) * 100

        return round(max(0, min(100, percentile)), 1)

    def _determine_risk_level(self, overall_score: float) -> str:
        """Determine overall risk level"""
        if overall_score >= 80:
            return 'Low'
        elif overall_score >= 65:
            return 'Moderate'
        elif overall_score >= 45:
            return 'High'
        else:
            return 'Critical'

    def _calculate_mobility_impact(
        self,
        medical_conditions: Dict,
        category_scores: Dict[str, float]
    ) -> float:
        """Calculate mobility impact score (0-100, higher = worse)"""

        mobility_impact = 0

        # High impact conditions
        high_impact_conditions = [
            'charcot', 'fracture', 'severe_arthritis', 'ulcer',
            'peripheral_arterial', 'severe_neuropathy'
        ]

        # Medium impact conditions
        medium_impact_conditions = [
            'hallux_rigidus', 'plantar_fasciitis', 'achilles',
            'severe_bunion', 'flat_foot'
        ]

        for cond_name, condition in medical_conditions.items():
            if hasattr(condition, 'detected') and condition.detected:
                severity = getattr(condition, 'severity', 'mild')

                if any(hc in cond_name.lower() for hc in high_impact_conditions):
                    if severity == 'severe':
                        mobility_impact += 25
                    elif severity == 'moderate':
                        mobility_impact += 15
                    else:
                        mobility_impact += 8

                elif any(mc in cond_name.lower() for mc in medium_impact_conditions):
                    if severity == 'severe':
                        mobility_impact += 15
                    elif severity == 'moderate':
                        mobility_impact += 10
                    else:
                        mobility_impact += 5

        # Factor in biomechanical and structural scores
        if category_scores.get('biomechanical_function', 100) < 60:
            mobility_impact += 10
        if category_scores.get('structural_integrity', 100) < 60:
            mobility_impact += 10

        return min(100, mobility_impact)

    def _calculate_fall_likelihood(
        self,
        medical_conditions: Dict,
        category_scores: Dict[str, float],
        risk_assessments: Optional[Dict]
    ) -> float:
        """Calculate fall likelihood (0-100 probability)"""

        # Start with base rate
        fall_risk = 15.0  # Base 15% for general population

        # If we have risk assessment data, use it
        if risk_assessments and 'fall_risk' in risk_assessments:
            fall_assessment = risk_assessments['fall_risk']
            if hasattr(fall_assessment, 'probability'):
                return fall_assessment.probability * 100

        # Otherwise estimate from conditions
        fall_risk_conditions = [
            'neuropathy', 'peripheral_arterial', 'balance', 'gait',
            'vision', 'weakness', 'dizziness'
        ]

        for cond_name, condition in medical_conditions.items():
            if any(frc in cond_name.lower() for frc in fall_risk_conditions):
                if hasattr(condition, 'detected') and condition.detected:
                    severity = getattr(condition, 'severity', 'mild')
                    if severity == 'severe':
                        fall_risk += 20
                    elif severity == 'moderate':
                        fall_risk += 12
                    else:
                        fall_risk += 6

        # Factor in biomechanical function
        biomech_score = category_scores.get('biomechanical_function', 100)
        if biomech_score < 50:
            fall_risk += 15
        elif biomech_score < 70:
            fall_risk += 8

        return min(100, round(fall_risk, 1))

    def _calculate_insurance_risk_factor(
        self,
        overall_score: float,
        mobility_impact: float,
        fall_likelihood: float,
        risk_assessments: Optional[Dict]
    ) -> float:
        """Calculate insurance risk multiplier (1.0 = average risk)"""

        # Start with baseline of 1.0
        risk_factor = 1.0

        # Adjust based on overall health score
        if overall_score < 40:
            risk_factor += 1.5
        elif overall_score < 60:
            risk_factor += 0.8
        elif overall_score < 75:
            risk_factor += 0.3
        elif overall_score > 90:
            risk_factor -= 0.2

        # Adjust based on mobility impact
        if mobility_impact > 70:
            risk_factor += 0.7
        elif mobility_impact > 50:
            risk_factor += 0.4
        elif mobility_impact > 30:
            risk_factor += 0.2

        # Adjust based on fall likelihood
        if fall_likelihood > 70:
            risk_factor += 0.8
        elif fall_likelihood > 50:
            risk_factor += 0.5
        elif fall_likelihood > 30:
            risk_factor += 0.2

        # Check for critical risk factors from assessments
        if risk_assessments:
            if 'diabetic_risk' in risk_assessments:
                diabetic_risk = risk_assessments['diabetic_risk']
                if hasattr(diabetic_risk, 'risk_level'):
                    if diabetic_risk.risk_level == 'critical':
                        risk_factor += 1.0
                    elif diabetic_risk.risk_level == 'high':
                        risk_factor += 0.5

        return round(max(0.5, min(5.0, risk_factor)), 2)

    def _identify_key_concerns(
        self,
        category_scores: Dict[str, float],
        medical_conditions: Dict
    ) -> List[str]:
        """Identify key health concerns"""
        concerns = []

        # Check category scores
        for category, score in category_scores.items():
            if score < 60:
                category_name = category.replace('_', ' ').title()
                concerns.append(f"Poor {category_name} (Score: {score:.0f}/100)")

        # Check for critical conditions
        critical_conditions = []
        for cond_name, condition in medical_conditions.items():
            if hasattr(condition, 'detected') and condition.detected:
                severity = getattr(condition, 'severity', 'mild')
                if severity in ['severe', 'critical']:
                    critical_conditions.append(
                        getattr(condition, 'condition_name', cond_name)
                    )

        if critical_conditions:
            concerns.extend(critical_conditions[:3])  # Top 3 critical conditions

        return concerns[:5]  # Return top 5 concerns

    def _identify_strengths(self, category_scores: Dict[str, float]) -> List[str]:
        """Identify health strengths"""
        strengths = []

        for category, score in category_scores.items():
            if score >= 85:
                category_name = category.replace('_', ' ').title()
                strengths.append(f"Excellent {category_name} (Score: {score:.0f}/100)")

        return strengths

    def _generate_recommendations(
        self,
        health_grade: str,
        key_concerns: List[str],
        risk_level: str
    ) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []

        # General recommendations based on health grade
        if health_grade in ['Critical', 'Poor']:
            recommendations.append("Immediate medical evaluation recommended")
            recommendations.append("Consider specialist referral")
        elif health_grade == 'Fair':
            recommendations.append("Schedule regular foot health monitoring")
            recommendations.append("Consider preventive interventions")
        else:
            recommendations.append("Continue current foot care routine")

        # Risk-based recommendations
        if risk_level in ['High', 'Critical']:
            recommendations.append("Implement fall prevention strategies")
            recommendations.append("Daily foot inspection recommended")

        # Concern-specific recommendations
        if any('circulatory' in str(c).lower() for c in key_concerns):
            recommendations.append("Vascular assessment recommended")

        if any('neurological' in str(c).lower() for c in key_concerns):
            recommendations.append("Neurological evaluation recommended")

        return recommendations[:5]

    def calculate_health_decline_rate(
        self,
        current_score: FootHealthScore,
        previous_scores: List[FootHealthScore]
    ) -> float:
        """Calculate rate of health decline over time"""

        if not previous_scores or len(previous_scores) < 1:
            return 0.0

        # Sort by timestamp
        all_scores = previous_scores + [current_score]
        all_scores.sort(key=lambda x: x.timestamp)

        # Calculate linear regression of score over time
        scores = [s.overall_score for s in all_scores]
        times = [(s.timestamp - all_scores[0].timestamp).days for s in all_scores]

        if len(set(times)) < 2:  # Need at least 2 different time points
            return 0.0

        # Simple linear regression
        n = len(scores)
        sum_x = sum(times)
        sum_y = sum(scores)
        sum_xy = sum(t * s for t, s in zip(times, scores))
        sum_xx = sum(t * t for t in times)

        # Slope (points per day)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)

        # Convert to points per year
        decline_rate = slope * 365

        return round(decline_rate, 2)

    def create_risk_matrix(
        self,
        medical_conditions: Dict,
        risk_assessments: Optional[Dict],
        foot_health_score: FootHealthScore
    ) -> RiskMatrix:
        """Create risk matrix with ONLY measurable metrics from 3D LiDAR scan"""

        # Structural Risk (measurable from 3D scan)
        structural_score = foot_health_score.category_scores.get('structural_integrity', 100)
        structural_risk = {
            'level': self._inverse_score_to_risk_level(structural_score),
            'score': 100 - structural_score,
            'factors': self._get_structural_factors(medical_conditions),
            'data_source': '3D LiDAR scan'
        }

        # Biomechanical Risk (measurable from 3D scan)
        biomech_score = foot_health_score.category_scores.get('biomechanical_function', 100)
        biomechanical_risk = {
            'level': self._inverse_score_to_risk_level(biomech_score),
            'score': 100 - biomech_score,
            'factors': self._get_biomechanical_factors(medical_conditions),
            'data_source': '3D LiDAR scan'
        }

        # Symmetry Risk (measurable from comparing left/right scans)
        symmetry_score = foot_health_score.category_scores.get('symmetry', 100)
        symmetry_risk = {
            'level': self._inverse_score_to_risk_level(symmetry_score),
            'score': 100 - symmetry_score,
            'factors': ['Significant left/right asymmetry'] if symmetry_score < 75 else [],
            'data_source': 'Left/Right scan comparison'
        }

        # Mobility Risk (inferred from structural + biomechanical)
        mobility_risk = {
            'level': self._score_to_risk_level(foot_health_score.mobility_impact_score),
            'score': foot_health_score.mobility_impact_score,
            'factors': foot_health_score.key_concerns,
            'data_source': 'Inferred from structural analysis'
        }

        # Fall Risk (inferred from biomechanical + structural)
        fall_risk = {
            'level': self._score_to_risk_level(foot_health_score.fall_likelihood),
            'score': foot_health_score.fall_likelihood,
            'factors': ['Poor balance indicators'] if biomech_score < 60 else [],
            'data_source': 'Inferred from gait/alignment analysis'
        }

        # Determine overall risk level (from measurable metrics only)
        risk_scores = [
            structural_risk['score'],
            biomechanical_risk['score'],
            symmetry_risk['score'],
            mobility_risk['score'],
            fall_risk['score']
        ]
        avg_risk_score = sum(risk_scores) / len(risk_scores)
        overall_risk_level = self._score_to_risk_level(avg_risk_score)

        # Identify critical flags (from measurable data only)
        critical_flags = []
        if structural_risk['level'] in ['High', 'Critical']:
            critical_flags.append('Severe structural deformities detected')
        if biomechanical_risk['level'] in ['High', 'Critical']:
            critical_flags.append('Significant biomechanical dysfunction')
        if fall_risk['score'] > 70:
            critical_flags.append('High fall risk indicated by gait analysis')
        if symmetry_risk['level'] in ['High', 'Critical']:
            critical_flags.append('Severe left/right asymmetry')

        return RiskMatrix(
            structural_risk=structural_risk,
            biomechanical_risk=biomechanical_risk,
            mobility_risk=mobility_risk,
            fall_risk=fall_risk,
            symmetry_risk=symmetry_risk,
            overall_risk_level=overall_risk_level,
            critical_flags=critical_flags
        )

    def _score_to_risk_level(self, score: float) -> str:
        """Convert a 0-100 score to risk level (higher score = higher risk)"""
        if score >= 70:
            return 'Critical'
        elif score >= 50:
            return 'High'
        elif score >= 30:
            return 'Moderate'
        else:
            return 'Low'

    def _inverse_score_to_risk_level(self, score: float) -> str:
        """Convert a 0-100 health score to risk level (lower score = higher risk)"""
        if score < 40:
            return 'Critical'
        elif score < 60:
            return 'High'
        elif score < 75:
            return 'Moderate'
        else:
            return 'Low'

    def _get_structural_factors(self, medical_conditions: Dict) -> List[str]:
        """Extract structural risk factors (measurable from 3D scan)"""
        factors = []
        structural_terms = ['hallux', 'bunion', 'hammer', 'claw', 'flat', 'cavus', 'arch', 'toe']

        for cond_name, condition in medical_conditions.items():
            if any(term in cond_name.lower() for term in structural_terms):
                if hasattr(condition, 'detected') and condition.detected:
                    factors.append(getattr(condition, 'condition_name', cond_name))

        return factors

    def _get_biomechanical_factors(self, medical_conditions: Dict) -> List[str]:
        """Extract biomechanical risk factors (measurable from 3D scan)"""
        factors = []
        biomech_terms = ['pronation', 'supination', 'gait', 'plantar', 'alignment', 'achilles']

        for cond_name, condition in medical_conditions.items():
            if any(term in cond_name.lower() for term in biomech_terms):
                if hasattr(condition, 'detected') and condition.detected:
                    factors.append(getattr(condition, 'condition_name', cond_name))

        return factors
