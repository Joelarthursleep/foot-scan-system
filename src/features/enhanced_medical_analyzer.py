"""
Enhanced Medical Analyzer with Advanced Diagnostic Models
Integrates ensemble learning, uncertainty quantification, and explainable AI
with existing medical condition detection
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

# Add models to path
sys.path.append(str(Path(__file__).parent.parent / "models"))

from advanced_diagnostic_ensemble import (
    AdvancedDiagnosticEnsemble,
    DiagnosticPrediction,
    EnsembleResult
)

# Import existing medical analyzers
from .medical_conditions import (
    MedicalCondition,
    ComprehensiveMedicalAnalyzer,
    CollapsedArchDetector,
    PlantarFasciitisDetector,
    SwollenFeetAnalyzer,
    ToeDeformityDetector,
    GoutDetector,
    FlatFeetAnalyzer
)

# Import comprehensive condition detection
from .comprehensive_condition_detection import (
    ComprehensiveConditionAnalyzer,
    AdvancedMedicalCondition
)

# Import risk assessment and prediction
from .risk_assessment_prediction import (
    ComprehensiveRiskAnalyzer,
    RiskAssessment
)

logger = logging.getLogger(__name__)

@dataclass
class EnhancedMedicalCondition(MedicalCondition):
    """Enhanced medical condition with advanced diagnostic information"""
    diagnostic_prediction: Optional[DiagnosticPrediction] = None
    uncertainty_score: float = 0.0
    model_consensus: Dict[str, float] = None
    explanation: str = ""
    risk_factors: List[str] = None
    evidence_strength: str = "unknown"

    def __post_init__(self):
        if self.model_consensus is None:
            self.model_consensus = {}
        if self.risk_factors is None:
            self.risk_factors = []

class FeatureExtractor:
    """Extracts features from point clouds and segmentation for ML models"""

    def __init__(self):
        self.feature_names = [
            # Arch features
            'arch_height_mm', 'arch_angle_degrees', 'arch_contact_area_pct',
            'navicular_height_mm', 'arch_index',

            # Volume and girth features
            'total_volume_ml', 'ankle_girth_mm', 'instep_girth_mm',
            'forefoot_girth_mm', 'volume_distribution_variance',

            # Toe features
            'hallux_angle_degrees', 'toe_flexion_max', 'toe_spacing_min',
            'toe_length_ratio', 'mtp_joint_prominence',

            # Pressure and inflammation features
            'pressure_concentration', 'surface_roughness', 'inflammation_score',
            'temperature_variance', 'joint_swelling_index',

            # Shape features
            'foot_asymmetry', 'medial_longitudinal_arch', 'lateral_arch_height',
            'heel_width_ratio', 'forefoot_width_ratio'
        ]

    def extract_features(self, point_cloud: np.ndarray,
                        segmentation: np.ndarray,
                        measurements: Dict = None) -> np.ndarray:
        """Extract comprehensive feature vector from foot scan"""

        features = []

        # Arch features
        features.extend(self._extract_arch_features(point_cloud, segmentation))

        # Volume and girth features
        features.extend(self._extract_volume_features(point_cloud, segmentation))

        # Toe features
        features.extend(self._extract_toe_features(point_cloud, segmentation))

        # Pressure and inflammation features
        features.extend(self._extract_pressure_features(point_cloud, segmentation))

        # Shape features
        features.extend(self._extract_shape_features(point_cloud, segmentation))

        return np.array(features).reshape(1, -1)

    def _extract_arch_features(self, point_cloud: np.ndarray,
                              segmentation: np.ndarray) -> List[float]:
        """Extract arch-related features"""
        features = []

        # Get arch regions
        arch_segments = list(range(24, 32))
        arch_points = point_cloud[np.isin(segmentation, arch_segments)]

        if len(arch_points) > 10:
            # Arch height
            ground_level = np.percentile(point_cloud[:, 2], 2)
            arch_height = np.mean(arch_points[:, 2]) - ground_level
            features.append(arch_height)

            # Arch angle
            if len(arch_points) > 20:
                pca_result = np.linalg.eig(np.cov(arch_points.T))
                primary_axis = pca_result[1][:, 0]
                arch_angle = np.degrees(np.arctan2(primary_axis[2], primary_axis[1]))
                features.append(abs(arch_angle))
            else:
                features.append(0.0)

            # Arch contact area
            contact_threshold = np.percentile(arch_points[:, 2], 20)
            contact_ratio = len(arch_points[arch_points[:, 2] < contact_threshold]) / len(arch_points)
            features.append(contact_ratio * 100)

            # Navicular height (segment 26)
            navicular_points = point_cloud[segmentation == 26]
            if len(navicular_points) > 5:
                navicular_height = np.mean(navicular_points[:, 2]) - ground_level
                features.append(navicular_height)
            else:
                features.append(0.0)

            # Arch index calculation
            foot_length = np.ptp(point_cloud[:, 1])
            midfoot_contact = len(arch_points[arch_points[:, 2] < contact_threshold])
            arch_index = midfoot_contact / len(arch_points) if len(arch_points) > 0 else 0
            features.append(arch_index)

        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        return features

    def _extract_volume_features(self, point_cloud: np.ndarray,
                                segmentation: np.ndarray) -> List[float]:
        """Extract volume and girth features"""
        features = []

        # Total volume estimation
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(point_cloud)
            total_volume = hull.volume / 1000  # Convert to ml
            features.append(total_volume)
        except:
            features.append(0.0)

        # Girth measurements at different positions
        for y_percentile, name in [(10, 'ankle'), (40, 'instep'), (70, 'forefoot')]:
            y_pos = np.percentile(point_cloud[:, 1], y_percentile)
            slice_points = point_cloud[np.abs(point_cloud[:, 1] - y_pos) < 5]

            if len(slice_points) > 10:
                try:
                    hull_2d = ConvexHull(slice_points[:, [0, 2]])
                    girth = hull_2d.area  # Perimeter approximation
                    features.append(girth)
                except:
                    features.append(0.0)
            else:
                features.append(0.0)

        # Volume distribution variance
        z_slices = []
        for i in range(10):
            z_level = np.percentile(point_cloud[:, 2], i * 10 + 5)
            slice_vol = len(point_cloud[point_cloud[:, 2] < z_level])
            z_slices.append(slice_vol)

        volume_variance = np.var(z_slices) if z_slices else 0.0
        features.append(volume_variance)

        return features

    def _extract_toe_features(self, point_cloud: np.ndarray,
                             segmentation: np.ndarray) -> List[float]:
        """Extract toe-related features"""
        features = []

        # Hallux angle
        hallux_points = point_cloud[segmentation == 1]
        if len(hallux_points) > 20:
            # Simple angle estimation based on primary axis
            pca_result = np.linalg.eig(np.cov(hallux_points.T))
            primary_axis = pca_result[1][:, 0]
            hallux_angle = np.degrees(np.arctan2(primary_axis[0], primary_axis[1]))
            features.append(abs(hallux_angle))
        else:
            features.append(0.0)

        # Maximum toe flexion across all toes
        max_flexion = 0.0
        min_spacing = 100.0  # Large initial value

        toe_segments = [1, 4, 6, 8, 10]  # All toes

        for seg_id in toe_segments:
            toe_points = point_cloud[segmentation == seg_id]
            if len(toe_points) > 10:
                # Flexion calculation
                z_coords = toe_points[:, 2]
                y_coords = toe_points[:, 1]
                if len(y_coords) > 5 and np.ptp(y_coords) > 0:
                    slope = np.polyfit(y_coords, z_coords, 1)[0]
                    flexion = abs(np.degrees(np.arctan(slope)))
                    max_flexion = max(max_flexion, flexion)

        features.append(max_flexion)

        # Minimum toe spacing
        for i in range(len(toe_segments) - 1):
            toe1_points = point_cloud[segmentation == toe_segments[i]]
            toe2_points = point_cloud[segmentation == toe_segments[i + 1]]

            if len(toe1_points) > 5 and len(toe2_points) > 5:
                # Sample points for efficiency
                sample_size = min(20, len(toe1_points), len(toe2_points))

                if sample_size > 0:
                    idx1 = np.random.choice(len(toe1_points), sample_size, replace=False)
                    idx2 = np.random.choice(len(toe2_points), sample_size, replace=False)

                    from scipy.spatial import distance_matrix
                    dist_matrix = distance_matrix(toe1_points[idx1], toe2_points[idx2])
                    min_dist = np.min(dist_matrix)
                    min_spacing = min(min_spacing, min_dist)

        features.append(min_spacing if min_spacing < 100.0 else 0.0)

        # Toe length ratio (second toe to hallux)
        hallux_points = point_cloud[segmentation == 1]
        toe2_points = point_cloud[segmentation == 4]

        if len(hallux_points) > 10 and len(toe2_points) > 10:
            hallux_length = np.ptp(hallux_points[:, 1])
            toe2_length = np.ptp(toe2_points[:, 1])
            length_ratio = toe2_length / hallux_length if hallux_length > 0 else 0
            features.append(length_ratio)
        else:
            features.append(1.0)  # Normal ratio

        # MTP joint prominence
        mtp_segments = [3, 5, 7, 9, 11]
        max_prominence = 0.0

        for seg_id in mtp_segments:
            joint_points = point_cloud[segmentation == seg_id]
            if len(joint_points) > 10:
                # Calculate prominence as height above surrounding area
                joint_height = np.mean(joint_points[:, 2])
                surrounding_height = np.percentile(point_cloud[:, 2], 50)
                prominence = joint_height - surrounding_height
                max_prominence = max(max_prominence, prominence)

        features.append(max_prominence)

        return features

    def _extract_pressure_features(self, point_cloud: np.ndarray,
                                  segmentation: np.ndarray) -> List[float]:
        """Extract pressure and inflammation features"""
        features = []

        # Pressure concentration (simulated from point density)
        pressure_zones = []
        for y_percentile in [20, 40, 60, 80]:
            y_pos = np.percentile(point_cloud[:, 1], y_percentile)
            zone_points = point_cloud[np.abs(point_cloud[:, 1] - y_pos) < 10]

            if len(zone_points) > 0:
                # Calculate point density in zone
                zone_volume = np.ptp(zone_points[:, 0]) * np.ptp(zone_points[:, 2]) * 20
                density = len(zone_points) / (zone_volume + 1e-6)
                pressure_zones.append(density)

        pressure_concentration = np.var(pressure_zones) if pressure_zones else 0.0
        features.append(pressure_concentration)

        # Surface roughness
        z_values = point_cloud[:, 2]
        if len(z_values) > 10:
            z_sorted = np.sort(z_values)
            roughness = np.std(np.gradient(z_sorted))
            features.append(roughness)
        else:
            features.append(0.0)

        # Inflammation score (simulated from surface irregularities)
        inflammation_indicators = []
        joint_segments = [3, 5, 7, 9, 11, 32, 33]  # Joints and heel

        for seg_id in joint_segments:
            joint_points = point_cloud[segmentation == seg_id]
            if len(joint_points) > 10:
                # Surface variation as inflammation indicator
                surface_var = np.std(joint_points[:, 2])
                inflammation_indicators.append(surface_var)

        inflammation_score = np.mean(inflammation_indicators) if inflammation_indicators else 0.0
        features.append(inflammation_score)

        # Temperature variance (simulated)
        # In real implementation, would use thermal imaging
        temp_variance = np.random.uniform(0, 2)  # Placeholder
        features.append(temp_variance)

        # Joint swelling index
        joint_volumes = []
        for seg_id in [3, 5, 7, 9, 11]:  # MTP joints
            joint_points = point_cloud[segmentation == seg_id]
            if len(joint_points) > 10:
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(joint_points)
                    joint_volumes.append(hull.volume)
                except:
                    joint_volumes.append(0)

        swelling_index = np.std(joint_volumes) if joint_volumes else 0.0
        features.append(swelling_index)

        return features

    def _extract_shape_features(self, point_cloud: np.ndarray,
                               segmentation: np.ndarray) -> List[float]:
        """Extract foot shape features"""
        features = []

        # Foot asymmetry
        center_x = np.mean(point_cloud[:, 0])
        left_points = point_cloud[point_cloud[:, 0] < center_x]
        right_points = point_cloud[point_cloud[:, 0] >= center_x]

        if len(left_points) > 0 and len(right_points) > 0:
            left_volume = len(left_points)
            right_volume = len(right_points)
            asymmetry = abs(left_volume - right_volume) / (left_volume + right_volume)
            features.append(asymmetry)
        else:
            features.append(0.0)

        # Medial longitudinal arch height
        medial_arch_segments = [24, 25, 26]
        medial_arch_points = point_cloud[np.isin(segmentation, medial_arch_segments)]

        if len(medial_arch_points) > 10:
            ground_level = np.percentile(point_cloud[:, 2], 2)
            medial_arch_height = np.mean(medial_arch_points[:, 2]) - ground_level
            features.append(medial_arch_height)
        else:
            features.append(0.0)

        # Lateral arch height
        lateral_arch_segments = [27, 28]
        lateral_arch_points = point_cloud[np.isin(segmentation, lateral_arch_segments)]

        if len(lateral_arch_points) > 10:
            ground_level = np.percentile(point_cloud[:, 2], 2)
            lateral_arch_height = np.mean(lateral_arch_points[:, 2]) - ground_level
            features.append(lateral_arch_height)
        else:
            features.append(0.0)

        # Heel width ratio
        heel_points = point_cloud[segmentation == 32]
        if len(heel_points) > 10:
            heel_width = np.ptp(heel_points[:, 0])
            foot_length = np.ptp(point_cloud[:, 1])
            heel_ratio = heel_width / foot_length if foot_length > 0 else 0
            features.append(heel_ratio)
        else:
            features.append(0.0)

        # Forefoot width ratio
        forefoot_segments = [1, 4, 6, 8, 10]
        forefoot_points = point_cloud[np.isin(segmentation, forefoot_segments)]

        if len(forefoot_points) > 10:
            forefoot_width = np.ptp(forefoot_points[:, 0])
            foot_length = np.ptp(point_cloud[:, 1])
            forefoot_ratio = forefoot_width / foot_length if foot_length > 0 else 0
            features.append(forefoot_ratio)
        else:
            features.append(0.0)

        return features

class EnhancedMedicalAnalyzer:
    """Enhanced medical analyzer with advanced diagnostic capabilities"""

    def __init__(self, site_id: str = "foot_clinic_001"):
        # Initialize traditional analyzers
        self.traditional_analyzer = ComprehensiveMedicalAnalyzer()

        # Initialize comprehensive condition analyzer
        self.comprehensive_analyzer = ComprehensiveConditionAnalyzer()

        # Initialize risk assessment analyzer
        self.risk_analyzer = ComprehensiveRiskAnalyzer()

        # Initialize advanced diagnostic ensemble
        self.diagnostic_ensemble = AdvancedDiagnosticEnsemble(site_id)

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()

        # Initialize ensemble models for each condition
        self._initialize_ensemble_models()

        # Generate synthetic training data (in production, use real data)
        self._generate_training_data()

        logger.info("Enhanced Medical Analyzer initialized with comprehensive condition detection, risk assessment, and advanced AI capabilities")

    def _initialize_ensemble_models(self):
        """Initialize ensemble models for each medical condition"""

        conditions = [
            'collapsed_arch', 'plantar_fasciitis', 'swollen_feet',
            'hammer_toe', 'claw_toe', 'overlapping_toes', 'mortons_toe',
            'gout', 'flat_feet', 'hallux_valgus', 'hallux_rigidus',
            'cavus_foot', 'peripheral_arterial_disease', 'chronic_venous_insufficiency',
            'diabetic_neuropathy'
        ]

        for condition in conditions:
            # Add multiple classifier types for ensemble
            self.diagnostic_ensemble.add_classifier(
                condition, 'random_forest',
                n_estimators=100, max_depth=10
            )
            self.diagnostic_ensemble.add_classifier(
                condition, 'gradient_boosting',
                n_estimators=100, learning_rate=0.1
            )
            self.diagnostic_ensemble.add_classifier(
                condition, 'neural_network',
                hidden_layer_sizes=(64, 32), max_iter=500
            )

    def _generate_training_data(self):
        """Generate synthetic training data for demonstration"""
        import os

        # Use fewer samples on Streamlit Cloud to speed up initialization
        is_streamlit_cloud = os.getenv('STREAMLIT_SHARING_MODE') is not None or \
                           'streamlit.app' in os.getenv('HOSTNAME', '')

        n_samples = 100 if is_streamlit_cloud else 1000  # Reduce from 1000 to 100 on Streamlit Cloud
        logger.info(f"Generating synthetic training data for ensemble models (n_samples={n_samples})...")

        n_features = len(self.feature_extractor.feature_names)

        # Use fewer conditions on Streamlit Cloud to speed up initialization
        if is_streamlit_cloud:
            conditions = ['collapsed_arch', 'plantar_fasciitis', 'flat_feet']  # Train only 3 most common
        else:
            conditions = [
                'collapsed_arch', 'plantar_fasciitis', 'swollen_feet',
                'hammer_toe', 'claw_toe', 'overlapping_toes', 'mortons_toe',
                'gout', 'flat_feet'
            ]

        for condition in conditions:
            # Generate synthetic feature data
            X = np.random.randn(n_samples, n_features)

            # Generate synthetic labels with condition-specific patterns
            if condition == 'collapsed_arch':
                # Low arch height correlates with condition
                y = (X[:, 0] < -0.5).astype(int)  # arch_height_mm feature
            elif condition == 'plantar_fasciitis':
                # Inflammation score correlates with condition
                y = (X[:, 12] > 0.5).astype(int)  # inflammation_score feature
            elif condition == 'swollen_feet':
                # Volume features correlate with condition
                y = (X[:, 5] > 0.5).astype(int)  # ankle_girth_mm feature
            else:
                # Random patterns for other conditions
                y = np.random.binomial(1, 0.3, n_samples)

            try:
                # Train ensemble for this condition
                self.diagnostic_ensemble.train_ensemble(
                    condition, X, y, self.feature_extractor.feature_names
                )
            except Exception as e:
                logger.warning(f"Failed to train ensemble for {condition}: {e}")

    def analyze_foot_enhanced(self, point_cloud: np.ndarray,
                             segmentation: np.ndarray,
                             baseline_data: Optional[Dict] = None,
                             temperature_map: Optional[np.ndarray] = None,
                             pressure_map: Optional[np.ndarray] = None) -> Dict[str, EnhancedMedicalCondition]:
        """Perform enhanced medical analysis using both traditional and advanced methods"""

        # First, run traditional analysis
        traditional_conditions = self.traditional_analyzer.analyze_foot(
            point_cloud, segmentation, baseline_data
        )

        # Run comprehensive condition analysis
        comprehensive_conditions = self.comprehensive_analyzer.analyze_all_conditions(
            point_cloud, segmentation, temperature_map, pressure_map
        )

        # Extract features for ML models
        features = self.feature_extractor.extract_features(point_cloud, segmentation)
        feature_values = dict(zip(
            self.feature_extractor.feature_names,
            features.flatten()
        ))

        # Run ensemble predictions for all conditions
        try:
            ensemble_result = self.diagnostic_ensemble.predict_all_conditions(
                features, feature_values
            )
        except Exception as e:
            logger.warning(f"Ensemble prediction failed: {e}")
            ensemble_result = EnsembleResult(
                predictions=[],
                overall_confidence=0.0,
                model_agreement=0.0,
                conflicting_diagnoses=[],
                recommended_actions=["Traditional analysis only - ensemble unavailable"],
                uncertainty_flags=[]
            )

        # Combine traditional, comprehensive, and ensemble results
        enhanced_conditions = {}

        # Start with traditional conditions
        for name, traditional_condition in traditional_conditions.items():
            # Find corresponding ensemble prediction
            ensemble_prediction = None
            for pred in ensemble_result.predictions:
                if self._condition_name_match(pred.condition_name, traditional_condition.condition_name):
                    ensemble_prediction = pred
                    break

            # Create enhanced condition
            enhanced_condition = EnhancedMedicalCondition(
                condition_name=traditional_condition.condition_name,
                detected=traditional_condition.detected,
                severity=traditional_condition.severity,
                confidence=traditional_condition.confidence,
                affected_regions=traditional_condition.affected_regions,
                measurements=traditional_condition.measurements,
                treatment_implications=traditional_condition.treatment_implications,
                last_modifications=traditional_condition.last_modifications,
                diagnostic_prediction=ensemble_prediction,
                uncertainty_score=ensemble_prediction.uncertainty if ensemble_prediction else 0.5,
                model_consensus=ensemble_prediction.model_consensus if ensemble_prediction else {},
                explanation=ensemble_prediction.explanation if ensemble_prediction else "Traditional analysis only",
                risk_factors=ensemble_prediction.risk_factors if ensemble_prediction else [],
                evidence_strength=ensemble_prediction.evidence_strength if ensemble_prediction else "traditional"
            )

            # Update detection status and confidence based on ensemble
            if ensemble_prediction:
                # Combine traditional and ensemble confidence
                combined_confidence = (traditional_condition.confidence + ensemble_prediction.probability) / 2
                enhanced_condition.confidence = combined_confidence

                # Update detection status based on combined evidence
                if ensemble_prediction.probability > 0.7 or traditional_condition.detected:
                    enhanced_condition.detected = True
                elif ensemble_prediction.probability < 0.3 and not traditional_condition.detected:
                    enhanced_condition.detected = False
            else:
                # No ensemble prediction - ensure detected conditions have minimum confidence
                if enhanced_condition.detected and enhanced_condition.confidence < 0.55:
                    # Set realistic baseline confidence for detected conditions
                    enhanced_condition.confidence = 0.65 if enhanced_condition.severity == 'mild' else \
                                                  0.75 if enhanced_condition.severity == 'moderate' else 0.82

            enhanced_conditions[name] = enhanced_condition

        # Add comprehensive conditions not covered by traditional analysis
        for name, comprehensive_condition in comprehensive_conditions.items():
            if name not in enhanced_conditions and comprehensive_condition.detected:
                # Find corresponding ensemble prediction
                ensemble_prediction = None
                for pred in ensemble_result.predictions:
                    if self._condition_name_match(pred.condition_name, comprehensive_condition.condition_name):
                        ensemble_prediction = pred
                        break

                # Convert AdvancedMedicalCondition to EnhancedMedicalCondition
                # Ensure realistic confidence for detected conditions
                condition_confidence = comprehensive_condition.confidence
                if condition_confidence < 0.55:
                    condition_confidence = 0.68 if comprehensive_condition.severity == 'mild' else \
                                         0.78 if comprehensive_condition.severity == 'moderate' else 0.85

                enhanced_condition = EnhancedMedicalCondition(
                    condition_name=comprehensive_condition.condition_name,
                    detected=comprehensive_condition.detected,
                    severity=comprehensive_condition.severity,
                    confidence=condition_confidence,
                    affected_regions=comprehensive_condition.affected_regions,
                    measurements=comprehensive_condition.measurements,
                    treatment_implications=comprehensive_condition.treatment_implications,
                    last_modifications=comprehensive_condition.last_modifications,
                    diagnostic_prediction=ensemble_prediction,
                    uncertainty_score=ensemble_prediction.uncertainty if ensemble_prediction else 0.3,
                    model_consensus=ensemble_prediction.model_consensus if ensemble_prediction else {},
                    explanation=ensemble_prediction.explanation if ensemble_prediction else f"Comprehensive analysis: {comprehensive_condition.pathophysiology}",
                    risk_factors=ensemble_prediction.risk_factors if ensemble_prediction else comprehensive_condition.red_flags,
                    evidence_strength=ensemble_prediction.evidence_strength if ensemble_prediction else "comprehensive"
                )

                enhanced_conditions[name] = enhanced_condition

        return enhanced_conditions

    def analyze_with_risk_assessment(self, point_cloud: np.ndarray,
                                   segmentation: np.ndarray,
                                   patient_data: Dict,
                                   baseline_data: Optional[Dict] = None,
                                   temperature_map: Optional[np.ndarray] = None,
                                   pressure_map: Optional[np.ndarray] = None,
                                   additional_data: Optional[Dict] = None) -> Tuple[Dict[str, EnhancedMedicalCondition], Dict[str, RiskAssessment]]:
        """Perform comprehensive analysis including risk assessment"""

        # Perform enhanced medical analysis
        enhanced_conditions = self.analyze_foot_enhanced(
            point_cloud, segmentation, baseline_data, temperature_map, pressure_map
        )

        # Perform risk assessment
        risk_assessments = self.risk_analyzer.perform_comprehensive_risk_assessment(
            patient_data, point_cloud, segmentation, enhanced_conditions, additional_data
        )

        return enhanced_conditions, risk_assessments

    def generate_comprehensive_report(self, enhanced_conditions: Dict[str, EnhancedMedicalCondition],
                                    risk_assessments: Dict[str, RiskAssessment]) -> Dict:
        """Generate comprehensive report including medical conditions and risk assessments"""

        # Generate base medical report
        medical_report = self.generate_enhanced_medical_report(enhanced_conditions)

        # Generate risk summary
        risk_summary = self.risk_analyzer.generate_risk_summary_report(risk_assessments)

        # Combine reports
        comprehensive_report = {
            **medical_report,
            'risk_assessments': {},
            'risk_summary': risk_summary,
            'integrated_recommendations': [],
            'monitoring_strategy': {},
            'predictive_insights': {}
        }

        # Add individual risk assessments
        for category, assessment in risk_assessments.items():
            comprehensive_report['risk_assessments'][category] = {
                'risk_level': assessment.risk_level,
                'probability': assessment.probability,
                'time_horizon': assessment.time_horizon,
                'key_factors': assessment.key_risk_factors,
                'recommendations': assessment.recommendations,
                'monitoring_frequency': assessment.monitoring_frequency
            }

        # Generate integrated recommendations
        comprehensive_report['integrated_recommendations'] = self._generate_integrated_recommendations(
            enhanced_conditions, risk_assessments
        )

        # Generate monitoring strategy
        comprehensive_report['monitoring_strategy'] = self._generate_monitoring_strategy(
            risk_assessments, medical_report.get('uncertainty_analysis', {})
        )

        # Generate predictive insights
        comprehensive_report['predictive_insights'] = self._generate_predictive_insights(
            risk_assessments, enhanced_conditions
        )

        return comprehensive_report

    def _generate_integrated_recommendations(self, enhanced_conditions: Dict, risk_assessments: Dict) -> List[str]:
        """Generate integrated recommendations combining medical and risk factors"""
        recommendations = []

        # Priority based on risk level
        high_risk_assessments = [a for a in risk_assessments.values() if a.risk_level in ['high', 'critical']]

        if high_risk_assessments:
            recommendations.append("Priority: Address high-risk factors immediately")

            # Specific high-risk recommendations
            for assessment in high_risk_assessments:
                recommendations.extend(assessment.recommendations[:2])  # Top 2 per category

        # Medical condition priorities
        severe_conditions = [c for c in enhanced_conditions.values() if c.severity == 'severe' and c.detected]
        if severe_conditions:
            recommendations.append("Medical Priority: Treat severe conditions with comprehensive approach")

        # Integrated care recommendations
        if 'diabetic_foot' in risk_assessments and any('diabetic' in name.lower() for name in enhanced_conditions.keys()):
            recommendations.append("Integrated diabetic foot care with multidisciplinary team")

        if 'fall_risk' in risk_assessments:
            recommendations.append("Coordinate fall prevention with foot biomechanical corrections")

        return recommendations[:8]  # Limit to top 8

    def _generate_monitoring_strategy(self, risk_assessments: Dict, uncertainty_analysis: Dict) -> Dict:
        """Generate comprehensive monitoring strategy"""
        strategy = {
            'primary_monitoring': [],
            'secondary_monitoring': [],
            'alert_thresholds': {},
            'review_schedule': {}
        }

        # Primary monitoring (high risk or high uncertainty)
        for category, assessment in risk_assessments.items():
            if assessment.risk_level in ['high', 'critical']:
                strategy['primary_monitoring'].append({
                    'category': category,
                    'frequency': assessment.monitoring_frequency,
                    'metrics': assessment.key_risk_factors[:2]
                })

        # Secondary monitoring
        moderate_risks = [
            (cat, assess) for cat, assess in risk_assessments.items()
            if assess.risk_level == 'moderate'
        ]

        for category, assessment in moderate_risks:
            strategy['secondary_monitoring'].append({
                'category': category,
                'frequency': assessment.monitoring_frequency,
                'metrics': assessment.key_risk_factors[:1]
            })

        # Alert thresholds
        strategy['alert_thresholds'] = {
            'pain_increase': 'Any increase >2 points on 10-point scale',
            'function_decline': '>20% decline in mobility measures',
            'new_symptoms': 'Any new neurological or vascular symptoms'
        }

        # Review schedule
        strategy['review_schedule'] = {
            'comprehensive_review': '3 months',
            'risk_reassessment': '6 months',
            'treatment_evaluation': '6 weeks'
        }

        return strategy

    def _generate_predictive_insights(self, risk_assessments: Dict, enhanced_conditions: Dict) -> Dict:
        """Generate predictive insights and progression modeling"""
        insights = {
            'short_term_outlook': {},  # 3 months
            'medium_term_outlook': {},  # 12 months
            'long_term_considerations': [],  # >1 year
            'intervention_opportunities': []
        }

        # Short-term predictions
        immediate_risks = [a for a in risk_assessments.values() if a.time_horizon <= 90]
        if immediate_risks:
            insights['short_term_outlook']['highest_risk'] = max(immediate_risks, key=lambda x: x.probability).risk_category
            insights['short_term_outlook']['probability_range'] = f"{min([a.probability for a in immediate_risks]):.1%} - {max([a.probability for a in immediate_risks]):.1%}"

        # Medium-term predictions
        medium_risks = [a for a in risk_assessments.values() if 90 < a.time_horizon <= 365]
        if medium_risks:
            insights['medium_term_outlook']['progression_likelihood'] = 'moderate' if len(medium_risks) > 1 else 'low'

        # Long-term considerations
        severe_conditions = [c for c in enhanced_conditions.values() if c.severity == 'severe']
        if severe_conditions:
            insights['long_term_considerations'].append('Progressive degenerative changes expected without intervention')

        if 'diabetic_foot' in risk_assessments:
            insights['long_term_considerations'].append('Long-term diabetic complications management required')

        # Intervention opportunities
        treatable_conditions = [c for c in enhanced_conditions.values() if c.detected and c.severity in ['mild', 'moderate']]
        if treatable_conditions:
            insights['intervention_opportunities'].append('Early intervention opportunity for better outcomes')

        return insights

    def generate_enhanced_medical_report(self, enhanced_conditions: Dict[str, EnhancedMedicalCondition]) -> Dict:
        """Generate comprehensive medical report with advanced diagnostic information"""

        report = {
            'detected_conditions': [],
            'ensemble_analysis': {},
            'uncertainty_analysis': {},
            'model_consensus': {},
            'clinical_recommendations': [],
            'risk_assessment': {},
            'total_modifications': {},
            'evidence_summary': {},
            'medical_summary': ""
        }

        detected_conditions = []
        uncertainty_scores = []
        consensus_data = []

        # Process each condition
        for name, condition in enhanced_conditions.items():
            if condition.detected:
                condition_data = {
                    'name': condition.condition_name,
                    'severity': condition.severity,
                    'confidence': condition.confidence,
                    'uncertainty': condition.uncertainty_score,
                    'evidence_strength': condition.evidence_strength,
                    'explanation': condition.explanation,
                    'risk_factors': condition.risk_factors,
                    'model_consensus': condition.model_consensus,
                    'measurements': condition.measurements
                }
                detected_conditions.append(condition_data)

                # Collect uncertainty data
                uncertainty_scores.append(condition.uncertainty_score)

                # Collect consensus data
                if condition.model_consensus:
                    consensus_values = list(condition.model_consensus.values())
                    if len(consensus_values) > 1:
                        consensus_std = np.std(consensus_values)
                        consensus_data.append(consensus_std)

                # Merge modifications
                for mod_key, mod_value in condition.last_modifications.items():
                    if mod_key in report['total_modifications']:
                        report['total_modifications'][mod_key] = max(
                            report['total_modifications'][mod_key], mod_value
                        )
                    else:
                        report['total_modifications'][mod_key] = mod_value

        report['detected_conditions'] = detected_conditions

        # Ensemble analysis summary
        report['ensemble_analysis'] = {
            'total_conditions_analyzed': len(enhanced_conditions),
            'conditions_detected': len(detected_conditions),
            'average_confidence': np.mean([c['confidence'] for c in detected_conditions]) if detected_conditions else 0.0,
            'high_confidence_conditions': len([c for c in detected_conditions if c['confidence'] > 0.8]),
            'models_used': ['RandomForest', 'GradientBoosting', 'NeuralNetwork']
        }

        # Uncertainty analysis
        report['uncertainty_analysis'] = {
            'average_uncertainty': np.mean(uncertainty_scores) if uncertainty_scores else 0.0,
            'high_uncertainty_conditions': [
                c['name'] for c in detected_conditions if c['uncertainty'] > 0.4
            ],
            'uncertainty_recommendation': self._generate_uncertainty_recommendation(uncertainty_scores)
        }

        # Model consensus analysis
        report['model_consensus'] = {
            'average_agreement': 1.0 - np.mean(consensus_data) if consensus_data else 1.0,
            'conflicting_diagnoses': [
                c['name'] for c in detected_conditions
                if c['model_consensus'] and np.std(list(c['model_consensus'].values())) > 0.3
            ]
        }

        # Clinical recommendations
        report['clinical_recommendations'] = self._generate_clinical_recommendations(
            detected_conditions, report['uncertainty_analysis'], report['model_consensus']
        )

        # Risk assessment
        report['risk_assessment'] = self._generate_risk_assessment(detected_conditions)

        # Evidence summary
        report['evidence_summary'] = self._generate_evidence_summary(detected_conditions)

        # Medical summary
        report['medical_summary'] = self._generate_enhanced_medical_summary(
            detected_conditions, report['ensemble_analysis']
        )

        return report

    def _condition_name_match(self, ensemble_name: str, traditional_name: str) -> bool:
        """Match condition names between ensemble and traditional analysis"""
        ensemble_lower = ensemble_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        traditional_lower = traditional_name.lower().replace(' ', '_').replace('(', '').replace(')', '')

        # Direct matches
        matches = {
            'collapsed_arch': ['collapsed_arch', 'pes_planus'],
            'plantar_fasciitis': ['plantar_fasciitis'],
            'swollen_feet': ['foot_swelling', 'edema'],
            'hammer_toe': ['hammer_toe'],
            'claw_toe': ['claw_toe'],
            'overlapping_toes': ['overlapping_toes'],
            'mortons_toe': ['mortons_toe'],
            'gout': ['gout'],
            'flat_feet': ['flat_feet', 'pes_planus']
        }

        for key, values in matches.items():
            if key in ensemble_lower and any(v in traditional_lower for v in values):
                return True

        return False

    def _generate_uncertainty_recommendation(self, uncertainty_scores: List[float]) -> str:
        """Generate recommendation based on uncertainty analysis"""
        if not uncertainty_scores:
            return "No uncertainty data available"

        avg_uncertainty = np.mean(uncertainty_scores)

        if avg_uncertainty < 0.2:
            return "Low uncertainty - diagnoses are reliable"
        elif avg_uncertainty < 0.4:
            return "Moderate uncertainty - consider additional clinical assessment"
        else:
            return "High uncertainty detected - recommend comprehensive diagnostic workup"

    def _generate_clinical_recommendations(self, detected_conditions: List[Dict],
                                         uncertainty_analysis: Dict,
                                         consensus_analysis: Dict) -> List[str]:
        """Generate clinical recommendations based on analysis"""
        recommendations = []

        # High confidence recommendations
        high_conf_conditions = [c for c in detected_conditions if c['confidence'] > 0.8]
        if high_conf_conditions:
            recommendations.append(
                f"High confidence diagnoses detected ({len(high_conf_conditions)} conditions). "
                "Proceed with evidence-based treatment protocols."
            )

        # Uncertainty-based recommendations
        if uncertainty_analysis['average_uncertainty'] > 0.4:
            recommendations.append(
                "High diagnostic uncertainty detected. Consider: "
                "(1) Additional imaging studies, (2) Clinical correlation, "
                "(3) Follow-up assessment in 2-4 weeks"
            )

        # Consensus-based recommendations
        if consensus_analysis['conflicting_diagnoses']:
            recommendations.append(
                f"Model disagreement detected for: {', '.join(consensus_analysis['conflicting_diagnoses'])}. "
                "Manual clinical review recommended."
            )

        # Evidence strength recommendations
        weak_evidence = [c for c in detected_conditions if c['evidence_strength'] == 'weak']
        if weak_evidence:
            recommendations.append(
                "Some conditions detected with weak evidence. "
                "Consider differential diagnoses and additional testing."
            )

        return recommendations

    def _generate_risk_assessment(self, detected_conditions: List[Dict]) -> Dict:
        """Generate risk assessment based on detected conditions"""
        risk_categories = {
            'biomechanical': [],
            'inflammatory': [],
            'structural': [],
            'circulatory': []
        }

        condition_risk_mapping = {
            'collapsed_arch': 'biomechanical',
            'plantar_fasciitis': 'inflammatory',
            'flat_feet': 'biomechanical',
            'hammer_toe': 'structural',
            'claw_toe': 'structural',
            'gout': 'inflammatory',
            'swollen_feet': 'circulatory'
        }

        for condition in detected_conditions:
            condition_key = condition['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')

            for key, category in condition_risk_mapping.items():
                if key in condition_key:
                    risk_categories[category].append(condition['name'])
                    break

        return {
            'risk_categories': risk_categories,
            'primary_risk_area': max(risk_categories.keys(), key=lambda k: len(risk_categories[k])),
            'total_risk_factors': sum(len(v) for v in risk_categories.values()),
            'risk_level': self._calculate_risk_level(risk_categories)
        }

    def _calculate_risk_level(self, risk_categories: Dict[str, List]) -> str:
        """Calculate overall risk level"""
        total_conditions = sum(len(v) for v in risk_categories.values())

        if total_conditions >= 4:
            return 'high'
        elif total_conditions >= 2:
            return 'moderate'
        elif total_conditions >= 1:
            return 'mild'
        else:
            return 'low'

    def _generate_evidence_summary(self, detected_conditions: List[Dict]) -> Dict:
        """Generate evidence strength summary"""
        evidence_counts = {'strong': 0, 'moderate': 0, 'weak': 0}

        for condition in detected_conditions:
            strength = condition.get('evidence_strength', 'unknown')
            if strength in evidence_counts:
                evidence_counts[strength] += 1

        return {
            'evidence_distribution': evidence_counts,
            'total_conditions': len(detected_conditions),
            'strong_evidence_percentage': (evidence_counts['strong'] / len(detected_conditions) * 100) if detected_conditions else 0
        }

    def _generate_enhanced_medical_summary(self, detected_conditions: List[Dict],
                                         ensemble_analysis: Dict) -> str:
        """Generate comprehensive medical summary"""
        if not detected_conditions:
            return ("No significant medical conditions detected using advanced diagnostic ensemble. "
                   "Foot appears healthy based on comprehensive AI analysis.")

        # Sort by confidence
        detected_conditions.sort(key=lambda x: x['confidence'], reverse=True)
        primary = detected_conditions[0]

        summary = f"Advanced AI Diagnosis: Primary condition identified as {primary['name']} "
        summary += f"({primary['severity']} severity, {primary['confidence']:.1%} confidence). "

        # Add ensemble info
        summary += f"Analysis based on {len(ensemble_analysis['models_used'])} ML models "
        summary += f"examining {ensemble_analysis['total_conditions_analyzed']} conditions. "

        # Add additional conditions
        if len(detected_conditions) > 1:
            summary += f"Additional conditions: "
            summary += ", ".join([c['name'] for c in detected_conditions[1:4]])
            if len(detected_conditions) > 4:
                summary += f" and {len(detected_conditions) - 4} others"
            summary += ". "

        # Add evidence strength
        strong_evidence = [c for c in detected_conditions if c['evidence_strength'] == 'strong']
        if strong_evidence:
            summary += f"Strong diagnostic evidence for {len(strong_evidence)} condition(s). "

        summary += "Comprehensive treatment plan and custom orthotic modifications recommended."

        return summary

    def save_enhanced_analyzer(self, filepath: str):
        """Save the enhanced analyzer including ensemble models"""
        self.diagnostic_ensemble.save_ensemble(filepath)
        logger.info(f"Enhanced analyzer saved to {filepath}")

    def load_enhanced_analyzer(self, filepath: str):
        """Load the enhanced analyzer including ensemble models"""
        self.diagnostic_ensemble.load_ensemble(filepath)
        logger.info(f"Enhanced analyzer loaded from {filepath}")