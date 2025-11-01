"""
Risk Assessment & Prediction System
Implements predictive models for diabetic foot screening, fall risk assessment,
sports injury prediction, progression modeling, and treatment response prediction
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import pickle

logger = logging.getLogger(__name__)

@dataclass
class RiskAssessment:
    """Risk assessment result"""
    risk_category: str  # 'diabetic_foot', 'fall_risk', 'sports_injury', etc.
    risk_level: str  # 'low', 'moderate', 'high', 'critical'
    risk_score: float  # 0-100 numerical score
    probability: float  # 0-1 probability of adverse outcome
    time_horizon: int  # Assessment period in days
    confidence_interval: Tuple[float, float]
    key_risk_factors: List[str]
    protective_factors: List[str]
    recommendations: List[str]
    monitoring_frequency: str  # 'daily', 'weekly', 'monthly', 'quarterly'
    predicted_progression: Optional[Dict] = None

@dataclass
class PredictionModel:
    """Base prediction model"""
    model_name: str
    model_type: str
    features: List[str]
    target_variable: str
    model: Any = None
    scaler: StandardScaler = field(default_factory=StandardScaler)
    is_trained: bool = False
    training_metrics: Dict = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

class DiabeticFootScreening:
    """Comprehensive diabetic foot risk screening"""

    def __init__(self):
        self.risk_factors = [
            'previous_ulceration', 'current_infection', 'neuropathy_severity',
            'vascular_insufficiency', 'foot_deformities', 'inappropriate_footwear',
            'poor_glycemic_control', 'visual_impairment', 'nephropathy',
            'limited_joint_mobility', 'callus_formation', 'pressure_points'
        ]

        # Initialize prediction models
        self.ulceration_model = PredictionModel(
            model_name="Ulceration Risk Predictor",
            model_type="classification",
            features=self.risk_factors,
            target_variable="ulcer_development"
        )

        self.healing_model = PredictionModel(
            model_name="Wound Healing Predictor",
            model_type="regression",
            features=self.risk_factors + ['wound_size', 'wound_depth', 'infection_present'],
            target_variable="healing_time_days"
        )

        # Train with synthetic data
        self._initialize_models()

    def screen_patient(self, patient_data: Dict,
                      point_cloud: np.ndarray,
                      segmentation: np.ndarray,
                      medical_conditions: Dict) -> RiskAssessment:
        """Perform comprehensive diabetic foot screening"""

        # Extract risk factors from various sources
        risk_factors = self._extract_risk_factors(patient_data, point_cloud, segmentation, medical_conditions)

        # Calculate base risk score
        base_risk_score = self._calculate_base_risk_score(risk_factors)

        # Apply machine learning prediction
        if self.ulceration_model.is_trained:
            feature_vector = self._prepare_feature_vector(risk_factors)
            ml_probability = self.ulceration_model.model.predict_proba(feature_vector)[0][1]
            ml_risk_score = ml_probability * 100
        else:
            ml_probability = base_risk_score / 100
            ml_risk_score = base_risk_score

        # Combine scores
        combined_score = (base_risk_score * 0.6 + ml_risk_score * 0.4)
        risk_level = self._classify_diabetic_risk_level(combined_score)

        # Calculate confidence interval
        ci = self._calculate_confidence_interval(ml_probability)

        # Identify key risk and protective factors
        key_risks = self._identify_key_risk_factors(risk_factors)
        protective_factors = self._identify_protective_factors(risk_factors)

        # Generate recommendations
        recommendations = self._generate_diabetic_recommendations(risk_level, key_risks)

        # Determine monitoring frequency
        monitoring = self._determine_monitoring_frequency(risk_level)

        # Predict progression if high risk
        progression = None
        if risk_level in ['high', 'critical']:
            progression = self._predict_diabetic_progression(risk_factors, medical_conditions)

        return RiskAssessment(
            risk_category='diabetic_foot',
            risk_level=risk_level,
            risk_score=combined_score,
            probability=ml_probability,
            time_horizon=365,  # 1 year assessment
            confidence_interval=ci,
            key_risk_factors=key_risks,
            protective_factors=protective_factors,
            recommendations=recommendations,
            monitoring_frequency=monitoring,
            predicted_progression=progression
        )

    def _extract_risk_factors(self, patient_data: Dict, point_cloud: np.ndarray,
                             segmentation: np.ndarray, medical_conditions: Dict) -> Dict[str, float]:
        """Extract diabetic foot risk factors"""

        factors = {}

        # Patient history factors
        factors['previous_ulceration'] = float(patient_data.get('history_of_ulcers', False))
        factors['current_infection'] = float(patient_data.get('active_infection', False))
        factors['poor_glycemic_control'] = float(patient_data.get('hba1c', 7) > 8)
        factors['visual_impairment'] = float(patient_data.get('visual_problems', False))
        factors['nephropathy'] = float(patient_data.get('kidney_disease', False))

        # Medical condition factors
        neuropathy_detected = any('neuropathy' in name.lower() for name, cond in medical_conditions.items() if cond.detected)
        factors['neuropathy_severity'] = 3.0 if neuropathy_detected else 1.0

        vascular_detected = any('arterial' in name.lower() or 'vascular' in name.lower()
                               for name, cond in medical_conditions.items() if cond.detected)
        factors['vascular_insufficiency'] = 3.0 if vascular_detected else 1.0

        # Foot deformity assessment
        structural_conditions = [name for name, cond in medical_conditions.items()
                               if cond.detected and any(term in name.lower() for term in
                               ['hallux', 'hammer', 'claw', 'cavus', 'flat'])]
        factors['foot_deformities'] = min(5.0, len(structural_conditions))

        # Pressure point analysis
        factors['pressure_points'] = self._assess_pressure_points(point_cloud, segmentation)

        # Callus formation (estimated from surface roughness)
        factors['callus_formation'] = self._assess_callus_formation(point_cloud)

        # Limited joint mobility (estimated from deformities)
        factors['limited_joint_mobility'] = min(3.0, factors['foot_deformities'] * 0.6)

        # Footwear assessment (simulated)
        factors['inappropriate_footwear'] = float(np.random.random() > 0.7)  # 30% inappropriate

        return factors

    def _calculate_base_risk_score(self, risk_factors: Dict[str, float]) -> float:
        """Calculate base diabetic foot risk score"""

        weights = {
            'previous_ulceration': 25,
            'current_infection': 20,
            'neuropathy_severity': 15,
            'vascular_insufficiency': 15,
            'foot_deformities': 10,
            'poor_glycemic_control': 8,
            'pressure_points': 5,
            'callus_formation': 3,
            'limited_joint_mobility': 2,
            'inappropriate_footwear': 2,
            'visual_impairment': 1,
            'nephropathy': 1
        }

        total_score = 0
        for factor, value in risk_factors.items():
            if factor in weights:
                total_score += weights[factor] * (value / 5.0)  # Normalize to 0-5 scale

        return min(100, total_score)

    def _prepare_feature_vector(self, risk_factors: Dict[str, float]) -> np.ndarray:
        """Prepare feature vector for ML model"""
        feature_vector = []
        for factor in self.risk_factors:
            feature_vector.append(risk_factors.get(factor, 0))

        return np.array(feature_vector).reshape(1, -1)

    def _classify_diabetic_risk_level(self, risk_score: float) -> str:
        """Classify diabetic foot risk level"""
        if risk_score < 20:
            return 'low'
        elif risk_score < 40:
            return 'moderate'
        elif risk_score < 70:
            return 'high'
        else:
            return 'critical'

    def _calculate_confidence_interval(self, probability: float) -> Tuple[float, float]:
        """Calculate confidence interval for prediction"""
        # Simplified CI calculation
        std_error = np.sqrt(probability * (1 - probability) / 100)
        ci_lower = max(0, probability - 1.96 * std_error)
        ci_upper = min(1, probability + 1.96 * std_error)
        return (ci_lower, ci_upper)

    def _identify_key_risk_factors(self, risk_factors: Dict[str, float]) -> List[str]:
        """Identify the most significant risk factors"""
        sorted_factors = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)
        return [factor for factor, value in sorted_factors[:5] if value > 2]

    def _identify_protective_factors(self, risk_factors: Dict[str, float]) -> List[str]:
        """Identify protective factors"""
        protective = []

        if risk_factors.get('poor_glycemic_control', 0) < 1:
            protective.append('Good glycemic control')

        if risk_factors.get('inappropriate_footwear', 0) < 1:
            protective.append('Appropriate footwear')

        if risk_factors.get('foot_deformities', 0) < 1:
            protective.append('No significant foot deformities')

        return protective

    def _generate_diabetic_recommendations(self, risk_level: str, key_risks: List[str]) -> List[str]:
        """Generate diabetic foot care recommendations"""
        recommendations = []

        if risk_level == 'critical':
            recommendations.extend([
                "Immediate podiatric consultation required",
                "Daily foot inspection mandatory",
                "Pressure-relieving footwear essential",
                "Consider total contact casting if appropriate"
            ])
        elif risk_level == 'high':
            recommendations.extend([
                "Weekly professional foot examination",
                "Daily self-foot inspection",
                "Therapeutic footwear prescription",
                "Quarterly vascular assessment"
            ])
        elif risk_level == 'moderate':
            recommendations.extend([
                "Monthly foot screening",
                "Daily foot self-examination",
                "Annual comprehensive foot evaluation",
                "Proper foot hygiene education"
            ])
        else:
            recommendations.extend([
                "Annual diabetic foot screening",
                "Basic foot care education",
                "Appropriate footwear guidance"
            ])

        # Risk-specific recommendations
        if 'neuropathy_severity' in key_risks:
            recommendations.append("Neuropathy management and monitoring")

        if 'vascular_insufficiency' in key_risks:
            recommendations.append("Vascular surgery consultation")

        if 'foot_deformities' in key_risks:
            recommendations.append("Orthopedic evaluation for deformity correction")

        return recommendations

    def _determine_monitoring_frequency(self, risk_level: str) -> str:
        """Determine monitoring frequency based on risk level"""
        frequency_map = {
            'low': 'annually',
            'moderate': 'quarterly',
            'high': 'monthly',
            'critical': 'weekly'
        }
        return frequency_map.get(risk_level, 'quarterly')

    def _predict_diabetic_progression(self, risk_factors: Dict, medical_conditions: Dict) -> Dict:
        """Predict diabetic foot condition progression"""
        progression = {
            'timeline': {},
            'key_milestones': [],
            'intervention_opportunities': []
        }

        # Simple progression model
        base_risk = sum(risk_factors.values()) / len(risk_factors)

        if base_risk > 3:
            progression['timeline']['3_months'] = 'High risk of callus/pressure ulcer development'
            progression['timeline']['6_months'] = 'Potential for minor tissue breakdown'
            progression['timeline']['12_months'] = 'Risk of major diabetic foot complications'

            progression['key_milestones'] = [
                'Formation of protective callus (4-8 weeks)',
                'Possible pre-ulcerative lesion (8-16 weeks)',
                'Ulcer development risk peak (16-24 weeks)'
            ]

            progression['intervention_opportunities'] = [
                'Immediate pressure redistribution',
                'Prophylactic debridement at 6 weeks',
                'Footwear modification at 12 weeks'
            ]

        return progression

    def _assess_pressure_points(self, point_cloud: np.ndarray, segmentation: np.ndarray) -> float:
        """Assess high-pressure points on foot"""
        pressure_score = 0

        # High-risk areas: metatarsal heads, heel, great toe
        high_risk_segments = [1, 3, 5, 7, 9, 11, 32]

        for seg in high_risk_segments:
            region_points = point_cloud[segmentation == seg]
            if len(region_points) > 5:
                # Check for prominence
                prominence = np.max(region_points[:, 2]) - np.percentile(point_cloud[:, 2], 25)
                if prominence > 6:  # Significant prominence
                    pressure_score += 1

        return min(5.0, pressure_score)

    def _assess_callus_formation(self, point_cloud: np.ndarray) -> float:
        """Assess callus formation from surface characteristics"""
        # Analyze surface roughness as proxy for callus formation
        z_values = point_cloud[:, 2]
        surface_roughness = np.std(np.gradient(np.sort(z_values)))

        # Normalize to 0-5 scale
        callus_score = min(5.0, surface_roughness)
        return callus_score

    def _initialize_models(self):
        """Initialize ML models with synthetic data"""
        try:
            # Generate synthetic training data
            n_samples = 1000
            n_features = len(self.risk_factors)

            # Ulceration prediction model
            X = np.random.randn(n_samples, n_features)
            # Create realistic ulcer risk: higher risk with more risk factors
            risk_sum = np.sum(np.maximum(0, X), axis=1)
            y_ulcer = (risk_sum > np.percentile(risk_sum, 70)).astype(int)

            # Train ulceration model
            X_train, X_test, y_train, y_test = train_test_split(X, y_ulcer, test_size=0.2, random_state=42)

            self.ulceration_model.model = LogisticRegression(random_state=42)
            self.ulceration_model.scaler.fit(X_train)
            X_train_scaled = self.ulceration_model.scaler.transform(X_train)
            X_test_scaled = self.ulceration_model.scaler.transform(X_test)

            self.ulceration_model.model.fit(X_train_scaled, y_train)
            self.ulceration_model.is_trained = True

            # Calculate training metrics
            train_score = self.ulceration_model.model.score(X_train_scaled, y_train)
            test_score = self.ulceration_model.model.score(X_test_scaled, y_test)

            self.ulceration_model.training_metrics = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'n_samples': n_samples
            }

            # Healing time model
            y_healing = np.maximum(1, risk_sum * 5 + np.random.normal(0, 10, n_samples))

            X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X, y_healing, test_size=0.2, random_state=42)

            self.healing_model.model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.healing_model.scaler.fit(X_train_h)
            X_train_h_scaled = self.healing_model.scaler.transform(X_train_h)
            X_test_h_scaled = self.healing_model.scaler.transform(X_test_h)

            self.healing_model.model.fit(X_train_h_scaled, y_train_h)
            self.healing_model.is_trained = True

            train_score_h = self.healing_model.model.score(X_train_h_scaled, y_train_h)
            test_score_h = self.healing_model.model.score(X_test_h_scaled, y_test_h)

            self.healing_model.training_metrics = {
                'train_r2': train_score_h,
                'test_r2': test_score_h,
                'n_samples': n_samples
            }

            logger.info(f"Diabetic foot models trained: Ulcer={test_score:.3f}, Healing={test_score_h:.3f}")

        except Exception as e:
            logger.warning(f"Failed to initialize diabetic foot models: {e}")

class FallRiskAssessment:
    """Fall risk assessment system"""

    def __init__(self):
        self.risk_factors = [
            'balance_impairment', 'gait_abnormality', 'muscle_weakness',
            'vision_problems', 'medications', 'cognitive_impairment',
            'environmental_hazards', 'previous_falls', 'fear_of_falling',
            'foot_problems', 'orthostatic_hypotension'
        ]

    def assess_fall_risk(self, patient_data: Dict, point_cloud: np.ndarray,
                        segmentation: np.ndarray, gait_data: Optional[Dict] = None) -> RiskAssessment:
        """Comprehensive fall risk assessment"""

        # Extract fall risk factors
        risk_factors = self._extract_fall_risk_factors(patient_data, point_cloud, segmentation, gait_data)

        # Calculate risk score
        risk_score = self._calculate_fall_risk_score(risk_factors)

        # Determine risk level
        risk_level = self._classify_fall_risk_level(risk_score)

        # Calculate fall probability
        fall_probability = self._calculate_fall_probability(risk_score)

        # Generate recommendations
        recommendations = self._generate_fall_prevention_recommendations(risk_level, risk_factors)

        # Identify key factors
        key_risks = [factor for factor, value in risk_factors.items() if value > 2]
        protective_factors = self._identify_fall_protective_factors(patient_data)

        return RiskAssessment(
            risk_category='fall_risk',
            risk_level=risk_level,
            risk_score=risk_score,
            probability=fall_probability,
            time_horizon=180,  # 6 months
            confidence_interval=(max(0, fall_probability - 0.1), min(1, fall_probability + 0.1)),
            key_risk_factors=key_risks,
            protective_factors=protective_factors,
            recommendations=recommendations,
            monitoring_frequency=self._determine_fall_monitoring_frequency(risk_level)
        )

    def _extract_fall_risk_factors(self, patient_data: Dict, point_cloud: np.ndarray,
                                  segmentation: np.ndarray, gait_data: Optional[Dict]) -> Dict[str, float]:
        """Extract fall risk factors"""

        factors = {}

        # Patient data factors
        factors['previous_falls'] = float(patient_data.get('fall_history', 0))
        factors['medications'] = min(5.0, patient_data.get('medication_count', 0) * 0.5)
        factors['vision_problems'] = float(patient_data.get('visual_impairment', False)) * 3
        factors['cognitive_impairment'] = float(patient_data.get('dementia_score', 0) > 2) * 3
        factors['fear_of_falling'] = float(patient_data.get('fall_anxiety', False)) * 2

        # Foot-related factors
        factors['foot_problems'] = self._assess_foot_fall_risk(point_cloud, segmentation)

        # Gait analysis
        if gait_data:
            factors['gait_abnormality'] = self._assess_gait_abnormality(gait_data)
            factors['balance_impairment'] = self._assess_balance_impairment(gait_data)
        else:
            # Estimate from foot structure
            factors['gait_abnormality'] = self._estimate_gait_issues(point_cloud, segmentation)
            factors['balance_impairment'] = factors['gait_abnormality'] * 0.8

        # Estimated factors (only if data available, otherwise default to low risk)
        age = patient_data.get('age', 45)  # Default to 45 instead of 65
        factors['muscle_weakness'] = float(age > 75) * 2 if age > 60 else 0
        factors['orthostatic_hypotension'] = float(patient_data.get('bp_medications', False)) * 1.5
        # Only add environmental hazards if there's evidence of high fall risk from other factors
        factors['environmental_hazards'] = 0.5  # Assume low environmental risk unless indicated otherwise

        return factors

    def _calculate_fall_risk_score(self, risk_factors: Dict[str, float]) -> float:
        """Calculate fall risk score"""

        weights = {
            'previous_falls': 20,
            'balance_impairment': 15,
            'gait_abnormality': 15,
            'muscle_weakness': 12,
            'foot_problems': 10,
            'medications': 8,
            'vision_problems': 8,
            'cognitive_impairment': 5,
            'fear_of_falling': 4,
            'orthostatic_hypotension': 2,
            'environmental_hazards': 1
        }

        total_score = 0
        for factor, value in risk_factors.items():
            if factor in weights:
                total_score += weights[factor] * min(1.0, value / 3.0)

        return min(100, total_score)

    def _classify_fall_risk_level(self, risk_score: float) -> str:
        """Classify fall risk level"""
        if risk_score < 25:
            return 'low'
        elif risk_score < 50:
            return 'moderate'
        elif risk_score < 75:
            return 'high'
        else:
            return 'critical'

    def _calculate_fall_probability(self, risk_score: float) -> float:
        """Calculate probability of fall within time horizon"""
        # Logistic function to convert score to probability
        return 1 / (1 + np.exp(-(risk_score - 50) / 15))

    def _assess_foot_fall_risk(self, point_cloud: np.ndarray, segmentation: np.ndarray) -> float:
        """Assess foot-related fall risk factors"""
        risk_score = 0

        # Check for balance-affecting deformities
        # Severe arch problems
        arch_points = point_cloud[np.isin(segmentation, range(24, 32))]
        if len(arch_points) > 20:
            arch_height = np.mean(arch_points[:, 2]) - np.percentile(point_cloud[:, 2], 10)
            if arch_height < 5 or arch_height > 25:  # Too flat or too high
                risk_score += 2

        # Toe deformities affecting balance
        toe_segments = [1, 4, 6, 8, 10]
        deformed_toes = 0
        for seg in toe_segments:
            toe_points = point_cloud[segmentation == seg]
            if len(toe_points) > 10:
                # Check for significant deformity
                z_range = np.ptp(toe_points[:, 2])
                if z_range > 8:  # Significant vertical deformity
                    deformed_toes += 1

        if deformed_toes >= 2:
            risk_score += 1.5

        # Heel instability
        heel_points = point_cloud[segmentation == 32]
        if len(heel_points) > 20:
            heel_width = np.ptp(heel_points[:, 0])
            if heel_width < 40:  # Narrow heel
                risk_score += 1

        return min(5.0, risk_score)

    def _assess_gait_abnormality(self, gait_data: Dict) -> float:
        """Assess gait abnormality from gait analysis data"""
        # Placeholder - would use real gait analysis
        abnormality_score = 0

        if gait_data.get('step_length_asymmetry', 0) > 10:
            abnormality_score += 2

        if gait_data.get('stride_variability', 0) > 15:
            abnormality_score += 2

        if gait_data.get('double_support_time', 0) > 25:
            abnormality_score += 1

        return min(5.0, abnormality_score)

    def _assess_balance_impairment(self, gait_data: Dict) -> float:
        """Assess balance impairment"""
        balance_score = 0

        if gait_data.get('mediolateral_sway', 0) > 5:
            balance_score += 2

        if gait_data.get('postural_control_score', 100) < 80:
            balance_score += 2

        return min(5.0, balance_score)

    def _estimate_gait_issues(self, point_cloud: np.ndarray, segmentation: np.ndarray) -> float:
        """Estimate gait issues from foot structure"""
        # Simplified estimation based on foot deformities
        gait_impact = 0

        # Severe forefoot deformities
        forefoot_segments = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for seg in forefoot_segments:
            region_points = point_cloud[segmentation == seg]
            if len(region_points) > 5:
                prominence = np.max(region_points[:, 2]) - np.median(region_points[:, 2])
                if prominence > 6:
                    gait_impact += 0.5

        return min(5.0, gait_impact)

    def _identify_fall_protective_factors(self, patient_data: Dict) -> List[str]:
        """Identify protective factors against falls"""
        protective = []

        if patient_data.get('exercise_regular', False):
            protective.append('Regular exercise program')

        if patient_data.get('vision_corrected', True):
            protective.append('Corrected vision')

        if patient_data.get('home_modifications', False):
            protective.append('Home safety modifications')

        return protective

    def _generate_fall_prevention_recommendations(self, risk_level: str, risk_factors: Dict) -> List[str]:
        """Generate fall prevention recommendations"""
        recommendations = []

        base_recommendations = {
            'low': [
                'Continue regular physical activity',
                'Annual vision screening',
                'Home safety assessment'
            ],
            'moderate': [
                'Balance and strength training program',
                'Medication review with physician',
                'Professional gait assessment',
                'Proper footwear evaluation'
            ],
            'high': [
                'Intensive balance rehabilitation',
                'Physical therapy consultation',
                'Assistive device assessment',
                'Monthly fall risk monitoring'
            ],
            'critical': [
                'Immediate fall prevention intervention',
                'Comprehensive geriatric assessment',
                'Consider supervised living environment',
                'Weekly monitoring and reassessment'
            ]
        }

        recommendations.extend(base_recommendations.get(risk_level, []))

        # Risk-specific recommendations
        if risk_factors.get('foot_problems', 0) > 2:
            recommendations.append('Podiatric evaluation and treatment')

        if risk_factors.get('gait_abnormality', 0) > 2:
            recommendations.append('Gait training and biomechanical correction')

        if risk_factors.get('balance_impairment', 0) > 2:
            recommendations.append('Balance-specific exercise program')

        return recommendations

    def _determine_fall_monitoring_frequency(self, risk_level: str) -> str:
        """Determine fall risk monitoring frequency"""
        frequency_map = {
            'low': 'annually',
            'moderate': 'semi-annually',
            'high': 'quarterly',
            'critical': 'monthly'
        }
        return frequency_map.get(risk_level, 'annually')

class SportsInjuryPredictor:
    """Sports injury prediction system"""

    def __init__(self):
        self.injury_types = [
            'ankle_sprain', 'plantar_fasciitis', 'stress_fracture',
            'achilles_tendinopathy', 'shin_splints', 'it_band_syndrome'
        ]

        # Initialize prediction models for each injury type
        self.models = {}
        for injury in self.injury_types:
            self.models[injury] = PredictionModel(
                model_name=f"{injury.replace('_', ' ').title()} Predictor",
                model_type="classification",
                features=['training_load', 'biomechanics_score', 'previous_injury', 'fatigue_level'],
                target_variable=f"{injury}_risk"
            )

    def predict_injury_risk(self, athlete_data: Dict, point_cloud: np.ndarray,
                           segmentation: np.ndarray, activity_data: Dict) -> Dict[str, RiskAssessment]:
        """Predict sports injury risk for multiple injury types"""

        predictions = {}

        for injury_type in self.injury_types:
            # Extract injury-specific risk factors
            risk_factors = self._extract_injury_risk_factors(
                injury_type, athlete_data, point_cloud, segmentation, activity_data
            )

            # Calculate risk score
            risk_score = self._calculate_injury_risk_score(injury_type, risk_factors)

            # Determine risk level
            risk_level = self._classify_injury_risk_level(risk_score)

            # Generate recommendations
            recommendations = self._generate_injury_prevention_recommendations(injury_type, risk_level)

            predictions[injury_type] = RiskAssessment(
                risk_category='sports_injury',
                risk_level=risk_level,
                risk_score=risk_score,
                probability=risk_score / 100,
                time_horizon=90,  # 3 months
                confidence_interval=(max(0, risk_score/100 - 0.15), min(1, risk_score/100 + 0.15)),
                key_risk_factors=list(risk_factors.keys()),
                protective_factors=self._identify_injury_protective_factors(athlete_data),
                recommendations=recommendations,
                monitoring_frequency='weekly'
            )

        return predictions

    def _extract_injury_risk_factors(self, injury_type: str, athlete_data: Dict,
                                   point_cloud: np.ndarray, segmentation: np.ndarray,
                                   activity_data: Dict) -> Dict[str, float]:
        """Extract injury-specific risk factors"""

        factors = {}

        # Common factors
        factors['training_load'] = activity_data.get('weekly_miles', 0) / 10
        factors['previous_injury'] = float(athlete_data.get(f'history_{injury_type}', False)) * 3
        factors['fatigue_level'] = activity_data.get('fatigue_score', 3)

        # Biomechanical factors from foot analysis
        if injury_type == 'ankle_sprain':
            factors['ankle_instability'] = self._assess_ankle_instability(point_cloud, segmentation)
            factors['proprioception_deficit'] = factors['previous_injury'] * 0.5

        elif injury_type == 'plantar_fasciitis':
            factors['arch_strain'] = self._assess_arch_strain_risk(point_cloud, segmentation)
            factors['calf_tightness'] = activity_data.get('calf_flexibility', 5)

        elif injury_type == 'stress_fracture':
            factors['bone_stress'] = self._assess_bone_stress_risk(activity_data, athlete_data)
            factors['footwear_mileage'] = activity_data.get('shoe_miles', 0) / 100

        # Sport-specific factors
        sport = athlete_data.get('sport', 'running')
        if sport in ['basketball', 'soccer']:
            factors['cutting_frequency'] = 4.0
        elif sport in ['running', 'marathon']:
            factors['repetitive_stress'] = factors['training_load'] * 1.2

        return factors

    def _calculate_injury_risk_score(self, injury_type: str, risk_factors: Dict) -> float:
        """Calculate injury-specific risk score"""

        # Injury-specific weights
        weights = {
            'ankle_sprain': {'previous_injury': 25, 'ankle_instability': 20, 'fatigue_level': 15},
            'plantar_fasciitis': {'arch_strain': 30, 'training_load': 20, 'calf_tightness': 15},
            'stress_fracture': {'bone_stress': 35, 'training_load': 25, 'footwear_mileage': 10}
        }

        injury_weights = weights.get(injury_type, {})
        total_score = 0

        for factor, value in risk_factors.items():
            weight = injury_weights.get(factor, 5)  # Default weight
            total_score += weight * min(1.0, value / 5.0)

        return min(100, total_score)

    def _classify_injury_risk_level(self, risk_score: float) -> str:
        """Classify injury risk level"""
        if risk_score < 30:
            return 'low'
        elif risk_score < 60:
            return 'moderate'
        else:
            return 'high'

    def _assess_ankle_instability(self, point_cloud: np.ndarray, segmentation: np.ndarray) -> float:
        """Assess ankle instability risk factors"""
        # Simplified assessment based on ankle geometry
        ankle_points = point_cloud[segmentation == 42]  # Ankle region

        if len(ankle_points) < 20:
            return 2.0  # Default moderate risk

        # Check for asymmetry or structural issues
        ankle_width = np.ptp(ankle_points[:, 0])
        if ankle_width < 50:  # Narrow ankle
            return 4.0

        return 2.0

    def _assess_arch_strain_risk(self, point_cloud: np.ndarray, segmentation: np.ndarray) -> float:
        """Assess plantar fascia strain risk"""
        arch_points = point_cloud[np.isin(segmentation, range(24, 32))]

        if len(arch_points) < 20:
            return 3.0

        # Calculate arch drop/height
        arch_height = np.mean(arch_points[:, 2]) - np.percentile(point_cloud[:, 2], 10)

        if arch_height < 8:  # Flat foot
            return 5.0
        elif arch_height > 25:  # High arch
            return 4.0

        return 2.0

    def _assess_bone_stress_risk(self, activity_data: Dict, athlete_data: Dict) -> float:
        """Assess bone stress fracture risk"""
        risk = 0

        # Training load factors
        if activity_data.get('weekly_miles', 0) > 50:
            risk += 2

        # Demographic factors
        if athlete_data.get('gender') == 'female':
            risk += 1

        if athlete_data.get('age', 25) > 40:
            risk += 1

        # Nutritional factors
        if athlete_data.get('calcium_intake', 1000) < 800:
            risk += 1

        return min(5.0, risk)

    def _identify_injury_protective_factors(self, athlete_data: Dict) -> List[str]:
        """Identify injury protective factors"""
        protective = []

        if athlete_data.get('strength_training', False):
            protective.append('Regular strength training')

        if athlete_data.get('warm_up_routine', False):
            protective.append('Consistent warm-up routine')

        if athlete_data.get('nutrition_optimal', False):
            protective.append('Optimal nutrition program')

        return protective

    def _generate_injury_prevention_recommendations(self, injury_type: str, risk_level: str) -> List[str]:
        """Generate injury-specific prevention recommendations"""

        base_recommendations = {
            'ankle_sprain': [
                'Proprioceptive balance training',
                'Ankle stability exercises',
                'Proper footwear selection'
            ],
            'plantar_fasciitis': [
                'Calf stretching program',
                'Arch support evaluation',
                'Gradual training progression'
            ],
            'stress_fracture': [
                'Bone density assessment',
                'Cross-training incorporation',
                'Nutrition optimization'
            ]
        }

        recommendations = base_recommendations.get(injury_type, ['General injury prevention'])

        if risk_level == 'high':
            recommendations.extend([
                'Professional biomechanical assessment',
                'Modified training schedule',
                'Weekly monitoring'
            ])

        return recommendations

class TreatmentResponsePredictor:
    """Treatment response prediction system"""

    def __init__(self):
        self.treatment_types = [
            'conservative_care', 'orthotic_therapy', 'physical_therapy',
            'surgical_intervention', 'injection_therapy'
        ]

    def predict_treatment_response(self, patient_data: Dict, condition_data: Dict,
                                 proposed_treatment: str) -> RiskAssessment:
        """Predict treatment response probability"""

        # Extract response factors
        response_factors = self._extract_response_factors(patient_data, condition_data, proposed_treatment)

        # Calculate success probability
        success_probability = self._calculate_success_probability(proposed_treatment, response_factors)

        # Estimate treatment timeline
        timeline = self._estimate_treatment_timeline(proposed_treatment, response_factors)

        # Generate monitoring recommendations
        monitoring = self._generate_treatment_monitoring(proposed_treatment, success_probability)

        return RiskAssessment(
            risk_category='treatment_response',
            risk_level=self._classify_response_likelihood(success_probability),
            risk_score=success_probability * 100,
            probability=success_probability,
            time_horizon=timeline,
            confidence_interval=(max(0, success_probability - 0.2), min(1, success_probability + 0.2)),
            key_risk_factors=self._identify_response_barriers(response_factors),
            protective_factors=self._identify_response_enhancers(response_factors),
            recommendations=monitoring,
            monitoring_frequency='bi-weekly'
        )

    def _extract_response_factors(self, patient_data: Dict, condition_data: Dict,
                                treatment: str) -> Dict[str, float]:
        """Extract factors affecting treatment response"""

        factors = {}

        # Patient factors
        factors['age'] = patient_data.get('age', 50) / 80  # Normalize
        factors['compliance'] = float(patient_data.get('medication_compliance', 0.8))
        factors['comorbidities'] = min(5.0, patient_data.get('comorbidity_count', 0))
        factors['bmi'] = max(1.0, patient_data.get('bmi', 25) / 25)

        # Condition factors
        factors['severity'] = self._map_severity_to_score(condition_data.get('severity', 'mild'))
        factors['duration'] = min(5.0, condition_data.get('duration_months', 3) / 12)
        factors['previous_treatments'] = min(3.0, condition_data.get('failed_treatments', 0))

        # Treatment-specific factors
        if treatment == 'surgical_intervention':
            factors['surgical_candidate'] = float(factors['age'] < 0.8 and factors['comorbidities'] < 3)
        elif treatment == 'orthotic_therapy':
            factors['foot_flexibility'] = 1.0 - factors['duration'] * 0.2

        return factors

    def _calculate_success_probability(self, treatment: str, factors: Dict) -> float:
        """Calculate treatment success probability"""

        # Base success rates (literature-based estimates)
        base_rates = {
            'conservative_care': 0.65,
            'orthotic_therapy': 0.75,
            'physical_therapy': 0.70,
            'surgical_intervention': 0.85,
            'injection_therapy': 0.60
        }

        base_prob = base_rates.get(treatment, 0.60)

        # Adjust for patient factors
        age_factor = 1.0 - (factors['age'] - 0.5) * 0.2
        severity_factor = 1.0 - factors['severity'] * 0.15
        compliance_factor = 0.5 + factors['compliance'] * 0.5

        adjusted_prob = base_prob * age_factor * severity_factor * compliance_factor

        return max(0.1, min(0.95, adjusted_prob))

    def _map_severity_to_score(self, severity: str) -> float:
        """Map severity string to numerical score"""
        severity_map = {
            'none': 0,
            'mild': 1,
            'moderate': 2,
            'severe': 3
        }
        return severity_map.get(severity.lower(), 2)

    def _estimate_treatment_timeline(self, treatment: str, factors: Dict) -> int:
        """Estimate treatment timeline in days"""

        base_timelines = {
            'conservative_care': 90,
            'orthotic_therapy': 60,
            'physical_therapy': 120,
            'surgical_intervention': 180,
            'injection_therapy': 30
        }

        base_timeline = base_timelines.get(treatment, 90)

        # Adjust for severity and other factors
        severity_multiplier = 1 + factors.get('severity', 2) * 0.2
        age_multiplier = 1 + factors.get('age', 0.5) * 0.3

        adjusted_timeline = base_timeline * severity_multiplier * age_multiplier

        return int(adjusted_timeline)

    def _classify_response_likelihood(self, probability: float) -> str:
        """Classify treatment response likelihood"""
        if probability > 0.8:
            return 'excellent'
        elif probability > 0.6:
            return 'good'
        elif probability > 0.4:
            return 'fair'
        else:
            return 'poor'

    def _identify_response_barriers(self, factors: Dict) -> List[str]:
        """Identify barriers to treatment response"""
        barriers = []

        if factors.get('age', 0) > 0.8:
            barriers.append('Advanced age')

        if factors.get('comorbidities', 0) > 2:
            barriers.append('Multiple comorbidities')

        if factors.get('compliance', 1) < 0.7:
            barriers.append('Poor compliance history')

        if factors.get('duration', 0) > 2:
            barriers.append('Chronic condition')

        return barriers

    def _identify_response_enhancers(self, factors: Dict) -> List[str]:
        """Identify factors enhancing treatment response"""
        enhancers = []

        if factors.get('compliance', 0) > 0.9:
            enhancers.append('Excellent compliance')

        if factors.get('age', 1) < 0.5:
            enhancers.append('Younger age')

        if factors.get('severity', 3) < 2:
            enhancers.append('Mild to moderate severity')

        return enhancers

    def _generate_treatment_monitoring(self, treatment: str, success_prob: float) -> List[str]:
        """Generate treatment monitoring recommendations"""

        monitoring = [
            f'Baseline assessment before {treatment}',
            'Regular progress evaluations',
            'Outcome measurement tracking'
        ]

        if success_prob < 0.6:
            monitoring.extend([
                'Close monitoring for poor response',
                'Early identification of treatment failure',
                'Alternative treatment planning'
            ])

        if treatment == 'surgical_intervention':
            monitoring.extend([
                'Post-operative complication monitoring',
                'Rehabilitation progress tracking'
            ])

        return monitoring

class ComprehensiveRiskAnalyzer:
    """Main risk assessment coordinator"""

    def __init__(self):
        self.diabetic_screener = DiabeticFootScreening()
        self.fall_assessor = FallRiskAssessment()
        self.injury_predictor = SportsInjuryPredictor()
        self.treatment_predictor = TreatmentResponsePredictor()

        logger.info("Comprehensive Risk Analyzer initialized")

    def perform_comprehensive_risk_assessment(self, patient_data: Dict,
                                            point_cloud: np.ndarray,
                                            segmentation: np.ndarray,
                                            medical_conditions: Dict,
                                            additional_data: Optional[Dict] = None) -> Dict[str, RiskAssessment]:
        """Perform comprehensive risk assessment across all categories"""

        assessments = {}

        # Diabetic foot screening (if diabetic)
        if patient_data.get('diabetes', False):
            try:
                assessments['diabetic_foot'] = self.diabetic_screener.screen_patient(
                    patient_data, point_cloud, segmentation, medical_conditions
                )
            except Exception as e:
                logger.warning(f"Diabetic foot screening failed: {e}")

        # Fall risk assessment (if elderly or indicated)
        if patient_data.get('age', 0) > 65 or patient_data.get('fall_risk_indicated', False):
            try:
                gait_data = additional_data.get('gait_analysis', {}) if additional_data else {}
                assessments['fall_risk'] = self.fall_assessor.assess_fall_risk(
                    patient_data, point_cloud, segmentation, gait_data
                )
            except Exception as e:
                logger.warning(f"Fall risk assessment failed: {e}")

        # Sports injury prediction (if athlete)
        if patient_data.get('athlete', False):
            try:
                activity_data = additional_data.get('activity_data', {}) if additional_data else {}
                injury_predictions = self.injury_predictor.predict_injury_risk(
                    patient_data, point_cloud, segmentation, activity_data
                )
                assessments.update(injury_predictions)
            except Exception as e:
                logger.warning(f"Sports injury prediction failed: {e}")

        # Treatment response prediction (if treatment planned)
        if additional_data and 'planned_treatment' in additional_data:
            try:
                condition_data = self._summarize_conditions(medical_conditions)
                assessments['treatment_response'] = self.treatment_predictor.predict_treatment_response(
                    patient_data, condition_data, additional_data['planned_treatment']
                )
            except Exception as e:
                logger.warning(f"Treatment response prediction failed: {e}")

        logger.info(f"Comprehensive risk assessment complete: {len(assessments)} categories evaluated")

        return assessments

    def _summarize_conditions(self, medical_conditions: Dict) -> Dict:
        """Summarize medical conditions for treatment prediction"""
        summary = {
            'detected_conditions': len([c for c in medical_conditions.values() if c.detected]),
            'severity': 'mild',
            'duration_months': 6  # Default
        }

        # Find most severe condition
        severities = [c.severity for c in medical_conditions.values() if c.detected]
        if 'severe' in severities:
            summary['severity'] = 'severe'
        elif 'moderate' in severities:
            summary['severity'] = 'moderate'

        return summary

    def generate_risk_summary_report(self, risk_assessments: Dict[str, RiskAssessment]) -> Dict:
        """Generate comprehensive risk summary report"""

        report = {
            'overall_risk_level': 'low',
            'priority_risks': [],
            'monitoring_plan': {},
            'intervention_recommendations': [],
            'risk_timeline': {},
            'confidence_summary': {}
        }

        # Determine overall risk level
        risk_levels = [assessment.risk_level for assessment in risk_assessments.values()]
        if 'critical' in risk_levels:
            report['overall_risk_level'] = 'critical'
        elif 'high' in risk_levels:
            report['overall_risk_level'] = 'high'
        elif 'moderate' in risk_levels:
            report['overall_risk_level'] = 'moderate'

        # Identify priority risks
        priority_risks = [
            (category, assessment) for category, assessment in risk_assessments.items()
            if assessment.risk_level in ['high', 'critical']
        ]

        report['priority_risks'] = [
            {
                'category': category,
                'risk_level': assessment.risk_level,
                'probability': assessment.probability,
                'key_factors': assessment.key_risk_factors[:3]
            }
            for category, assessment in priority_risks
        ]

        # Create monitoring plan
        for category, assessment in risk_assessments.items():
            report['monitoring_plan'][category] = {
                'frequency': assessment.monitoring_frequency,
                'key_metrics': assessment.key_risk_factors[:2],
                'timeline': assessment.time_horizon
            }

        # Collect intervention recommendations
        all_recommendations = []
        for assessment in risk_assessments.values():
            all_recommendations.extend(assessment.recommendations)

        # Remove duplicates and prioritize
        unique_recommendations = list(set(all_recommendations))
        report['intervention_recommendations'] = unique_recommendations[:10]  # Top 10

        # Risk timeline
        report['risk_timeline'] = {
            category: f"{assessment.time_horizon} days"
            for category, assessment in risk_assessments.items()
        }

        # Confidence summary
        report['confidence_summary'] = {
            'average_confidence': np.mean([
                (assessment.confidence_interval[0] + assessment.confidence_interval[1]) / 2
                for assessment in risk_assessments.values()
            ]),
            'high_confidence_predictions': len([
                assessment for assessment in risk_assessments.values()
                if (assessment.confidence_interval[1] - assessment.confidence_interval[0]) < 0.3
            ])
        }

        return report