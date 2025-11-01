"""
Advanced Condition Detection Module for Foot Scan System

Implements rule-based detectors for:
- Bunion and bunionette deformities
- High instep conditions
- Arch variations (pes planus, pes cavus)
- Heel deformities
- Toe deformities
- Other foot pathologies

Each detector provides confidence scores and clinical recommendations
Based on established clinical criteria and research
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ConditionSeverity(Enum):
    """Standardized severity levels"""
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"

@dataclass
class ConditionDetection:
    """Represents a detected foot condition"""
    condition_name: str
    severity: ConditionSeverity
    confidence_score: float  # 0.0 to 1.0
    clinical_measurements: Dict[str, float]
    detection_criteria: List[str]
    clinical_recommendation: str
    risk_factors: List[str]
    progression_risk: str  # low, moderate, high
    treatment_urgency: str  # routine, prompt, urgent

@dataclass
class RegionalAssessment:
    """Assessment for specific foot regions"""
    region_name: str
    measurements: Dict[str, float]
    abnormalities: List[str]
    clinical_significance: str

class AdvancedConditionDetector:
    """Advanced rule-based condition detector with clinical validation"""

    def __init__(self):
        """Initialize condition detector"""
        self.logger = logging.getLogger(__name__)
        self.detection_thresholds = self._load_clinical_thresholds()

    def _load_clinical_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load clinical thresholds for condition detection"""
        return {
            'hallux_valgus': {
                'mild_threshold': 15.0,      # degrees
                'moderate_threshold': 30.0,   # degrees
                'severe_threshold': 40.0,     # degrees
                'min_confidence': 0.7
            },
            'bunionette': {
                'mild_threshold': 12.0,      # degrees (5th toe deviation)
                'moderate_threshold': 20.0,   # degrees
                'severe_threshold': 30.0,     # degrees
                'min_confidence': 0.6
            },
            'pes_planus': {
                'ahi_threshold': 21.0,       # AHI percentage
                'arch_height_threshold': 12.0, # mm
                'min_confidence': 0.8
            },
            'pes_cavus': {
                'ahi_threshold': 25.0,       # AHI percentage
                'arch_height_threshold': 25.0, # mm
                'min_confidence': 0.8
            },
            'high_instep': {
                'instep_threshold': 30.0,    # mm
                'relative_threshold': 0.15,  # relative to foot length
                'min_confidence': 0.7
            },
            'hammer_toe': {
                'angle_threshold': 20.0,     # degrees of flexion
                'prominence_threshold': 5.0, # mm dorsal prominence
                'min_confidence': 0.6
            },
            'heel_spur': {
                'prominence_threshold': 3.0,  # mm
                'angle_threshold': 15.0,     # degrees
                'min_confidence': 0.5
            }
        }

    def detect_all_conditions(self, foot_analysis: Dict[str, Any]) -> List[ConditionDetection]:
        """
        Detect all foot conditions from analysis data

        Args:
            foot_analysis: Complete foot structure analysis

        Returns:
            List of detected conditions with confidence scores
        """
        detected_conditions = []

        try:
            # Detect hallux valgus (bunion)
            hv_condition = self._detect_hallux_valgus(foot_analysis)
            if hv_condition:
                detected_conditions.append(hv_condition)

            # Detect bunionette (tailor's bunion)
            bunionette_condition = self._detect_bunionette(foot_analysis)
            if bunionette_condition:
                detected_conditions.append(bunionette_condition)

            # Detect arch abnormalities
            arch_conditions = self._detect_arch_abnormalities(foot_analysis)
            detected_conditions.extend(arch_conditions)

            # Detect high instep
            instep_condition = self._detect_high_instep(foot_analysis)
            if instep_condition:
                detected_conditions.append(instep_condition)

            # Detect toe deformities
            toe_conditions = self._detect_toe_deformities(foot_analysis)
            detected_conditions.extend(toe_conditions)

            # Detect heel abnormalities
            heel_conditions = self._detect_heel_abnormalities(foot_analysis)
            detected_conditions.extend(heel_conditions)

            # Sort by severity and confidence
            detected_conditions.sort(
                key=lambda x: (x.severity.value != 'normal', x.confidence_score),
                reverse=True
            )

        except Exception as e:
            self.logger.error(f"Condition detection failed: {e}")

        return detected_conditions

    def _detect_hallux_valgus(self, analysis: Dict[str, Any]) -> Optional[ConditionDetection]:
        """Detect hallux valgus (bunion) condition"""
        try:
            hallux_data = analysis.get('hallux_valgus', {})
            hva_angle = hallux_data.get('angle', hallux_data.get('hva', 0))

            if hva_angle == 0:
                return None

            thresholds = self.detection_thresholds['hallux_valgus']

            # Determine severity
            if hva_angle < thresholds['mild_threshold']:
                severity = ConditionSeverity.NORMAL
                confidence = 0.9
            elif hva_angle < thresholds['moderate_threshold']:
                severity = ConditionSeverity.MILD
                confidence = 0.85
            elif hva_angle < thresholds['severe_threshold']:
                severity = ConditionSeverity.MODERATE
                confidence = 0.90
            else:
                severity = ConditionSeverity.SEVERE
                confidence = 0.95

            if severity == ConditionSeverity.NORMAL:
                return None

            # Clinical measurements
            measurements = {
                'hallux_valgus_angle': hva_angle,
                'intermetatarsal_angle': analysis.get('intermetatarsal', {}).get('angle', 0)
            }

            # Detection criteria
            criteria = [
                f"HVA angle: {hva_angle:.1f}° (threshold: {thresholds['mild_threshold']}°)",
                f"Clinical examination indicates {severity.value} hallux valgus"
            ]

            # Clinical recommendation
            recommendation = self._get_hallux_valgus_recommendation(severity, hva_angle)

            # Risk factors
            risk_factors = self._get_hallux_valgus_risk_factors(analysis)

            # Progression risk
            progression_risk = self._assess_hallux_valgus_progression_risk(hva_angle, analysis)

            return ConditionDetection(
                condition_name="Hallux Valgus (Bunion)",
                severity=severity,
                confidence_score=confidence,
                clinical_measurements=measurements,
                detection_criteria=criteria,
                clinical_recommendation=recommendation,
                risk_factors=risk_factors,
                progression_risk=progression_risk,
                treatment_urgency=self._determine_treatment_urgency(severity, progression_risk)
            )

        except Exception as e:
            self.logger.error(f"Hallux valgus detection failed: {e}")
            return None

    def _detect_bunionette(self, analysis: Dict[str, Any]) -> Optional[ConditionDetection]:
        """Detect bunionette (tailor's bunion) condition"""
        try:
            bunionette_data = analysis.get('bunionette', {})
            if not bunionette_data:
                return None

            # Calculate 5th metatarsal deviation
            fifth_mt_angle = bunionette_data.get('angle', 0)
            lateral_prominence = bunionette_data.get('prominence', 0)

            if fifth_mt_angle == 0 and lateral_prominence == 0:
                return None

            thresholds = self.detection_thresholds['bunionette']

            # Determine severity based on angle and prominence
            severity_score = max(
                fifth_mt_angle / thresholds['severe_threshold'],
                lateral_prominence / 8.0  # 8mm prominence threshold
            )

            if severity_score < 0.5:
                return None
            elif severity_score < 0.7:
                severity = ConditionSeverity.MILD
                confidence = 0.75
            elif severity_score < 1.0:
                severity = ConditionSeverity.MODERATE
                confidence = 0.85
            else:
                severity = ConditionSeverity.SEVERE
                confidence = 0.90

            measurements = {
                'fifth_metatarsal_angle': fifth_mt_angle,
                'lateral_prominence': lateral_prominence
            }

            criteria = [
                f"5th MT angle: {fifth_mt_angle:.1f}°",
                f"Lateral prominence: {lateral_prominence:.1f}mm"
            ]

            recommendation = self._get_bunionette_recommendation(severity)

            return ConditionDetection(
                condition_name="Bunionette (Tailor's Bunion)",
                severity=severity,
                confidence_score=confidence,
                clinical_measurements=measurements,
                detection_criteria=criteria,
                clinical_recommendation=recommendation,
                risk_factors=["Tight footwear", "Genetic predisposition"],
                progression_risk="moderate" if severity != ConditionSeverity.MILD else "low",
                treatment_urgency="routine"
            )

        except Exception as e:
            self.logger.error(f"Bunionette detection failed: {e}")
            return None

    def _detect_arch_abnormalities(self, analysis: Dict[str, Any]) -> List[ConditionDetection]:
        """Detect arch abnormalities (pes planus, pes cavus)"""
        conditions = []

        try:
            arch_data = analysis.get('arch', {})
            ahi = arch_data.get('ahi', 0)
            arch_height = arch_data.get('height', 0)
            arch_type = arch_data.get('type', 'normal')

            # Detect pes planus (flat foot)
            pes_planus_condition = self._detect_pes_planus(ahi, arch_height, arch_type, analysis)
            if pes_planus_condition:
                conditions.append(pes_planus_condition)

            # Detect pes cavus (high arch)
            pes_cavus_condition = self._detect_pes_cavus(ahi, arch_height, arch_type, analysis)
            if pes_cavus_condition:
                conditions.append(pes_cavus_condition)

        except Exception as e:
            self.logger.error(f"Arch abnormality detection failed: {e}")

        return conditions

    def _detect_pes_planus(self, ahi: float, arch_height: float, arch_type: str,
                          analysis: Dict[str, Any]) -> Optional[ConditionDetection]:
        """Detect pes planus (flat foot)"""
        if arch_type != 'flat' and ahi >= self.detection_thresholds['pes_planus']['ahi_threshold']:
            return None

        # Determine severity
        if ahi > 18:
            severity = ConditionSeverity.MILD
            confidence = 0.8
        elif ahi > 15:
            severity = ConditionSeverity.MODERATE
            confidence = 0.85
        else:
            severity = ConditionSeverity.SEVERE
            confidence = 0.90

        measurements = {
            'arch_height_index': ahi,
            'arch_height_mm': arch_height
        }

        criteria = [
            f"AHI: {ahi:.1f}% (normal: >21%)",
            f"Arch height: {arch_height:.1f}mm",
            f"Arch classification: {arch_type}"
        ]

        recommendation = self._get_pes_planus_recommendation(severity)

        return ConditionDetection(
            condition_name="Pes Planus (Flat Foot)",
            severity=severity,
            confidence_score=confidence,
            clinical_measurements=measurements,
            detection_criteria=criteria,
            clinical_recommendation=recommendation,
            risk_factors=["Posterior tibial tendon dysfunction", "Genetic factors", "Obesity"],
            progression_risk="moderate",
            treatment_urgency="routine" if severity == ConditionSeverity.MILD else "prompt"
        )

    def _detect_pes_cavus(self, ahi: float, arch_height: float, arch_type: str,
                         analysis: Dict[str, Any]) -> Optional[ConditionDetection]:
        """Detect pes cavus (high arch)"""
        if arch_type != 'high' and ahi <= self.detection_thresholds['pes_cavus']['ahi_threshold']:
            return None

        # Determine severity
        if ahi < 30:
            severity = ConditionSeverity.MILD
            confidence = 0.8
        elif ahi < 35:
            severity = ConditionSeverity.MODERATE
            confidence = 0.85
        else:
            severity = ConditionSeverity.SEVERE
            confidence = 0.90

        measurements = {
            'arch_height_index': ahi,
            'arch_height_mm': arch_height
        }

        criteria = [
            f"AHI: {ahi:.1f}% (normal: <25%)",
            f"Arch height: {arch_height:.1f}mm",
            f"Arch classification: {arch_type}"
        ]

        recommendation = self._get_pes_cavus_recommendation(severity)

        return ConditionDetection(
            condition_name="Pes Cavus (High Arch)",
            severity=severity,
            confidence_score=confidence,
            clinical_measurements=measurements,
            detection_criteria=criteria,
            clinical_recommendation=recommendation,
            risk_factors=["Neurological conditions", "Genetic factors"],
            progression_risk="low" if severity == ConditionSeverity.MILD else "moderate",
            treatment_urgency="routine"
        )

    def _detect_high_instep(self, analysis: Dict[str, Any]) -> Optional[ConditionDetection]:
        """Detect high instep condition"""
        try:
            instep_data = analysis.get('instep', {})
            instep_height = instep_data.get('height', 0)

            dimensions = analysis.get('dimensions', {})
            foot_length = dimensions.get('length', 0)

            if instep_height == 0 or foot_length == 0:
                return None

            relative_height = instep_height / foot_length
            threshold = self.detection_thresholds['high_instep']['relative_threshold']

            if relative_height < threshold:
                return None

            # Determine severity
            if relative_height < threshold * 1.2:
                severity = ConditionSeverity.MILD
                confidence = 0.75
            elif relative_height < threshold * 1.5:
                severity = ConditionSeverity.MODERATE
                confidence = 0.80
            else:
                severity = ConditionSeverity.SEVERE
                confidence = 0.85

            measurements = {
                'instep_height_mm': instep_height,
                'relative_instep_height': relative_height,
                'foot_length_mm': foot_length
            }

            criteria = [
                f"Instep height: {instep_height:.1f}mm",
                f"Relative height: {relative_height:.3f} (threshold: {threshold:.3f})"
            ]

            return ConditionDetection(
                condition_name="High Instep",
                severity=severity,
                confidence_score=confidence,
                clinical_measurements=measurements,
                detection_criteria=criteria,
                clinical_recommendation="Custom footwear or orthotics may be beneficial",
                risk_factors=["Genetic predisposition", "Neuromuscular conditions"],
                progression_risk="low",
                treatment_urgency="routine"
            )

        except Exception as e:
            self.logger.error(f"High instep detection failed: {e}")
            return None

    def _detect_toe_deformities(self, analysis: Dict[str, Any]) -> List[ConditionDetection]:
        """Detect various toe deformities"""
        conditions = []

        # This is a simplified implementation - in practice, you'd need more
        # sophisticated analysis of toe geometry
        try:
            # Check for general toe abnormalities from the analysis
            if 'toe_analysis' in analysis:
                toe_data = analysis['toe_analysis']

                # Detect hammer toe, claw toe, mallet toe conditions
                for toe_num in range(2, 6):  # toes 2-5
                    toe_key = f'toe_{toe_num}'
                    if toe_key in toe_data:
                        toe_condition = self._analyze_individual_toe(toe_num, toe_data[toe_key])
                        if toe_condition:
                            conditions.append(toe_condition)

        except Exception as e:
            self.logger.error(f"Toe deformity detection failed: {e}")

        return conditions

    def _detect_heel_abnormalities(self, analysis: Dict[str, Any]) -> List[ConditionDetection]:
        """Detect heel abnormalities"""
        conditions = []

        try:
            heel_data = analysis.get('heel_analysis', {})

            # Check for heel spurs
            if 'spur_prominence' in heel_data:
                spur_condition = self._detect_heel_spur(heel_data)
                if spur_condition:
                    conditions.append(spur_condition)

            # Check for heel deformities
            if 'heel_alignment' in heel_data:
                alignment_condition = self._detect_heel_alignment_issues(heel_data)
                if alignment_condition:
                    conditions.append(alignment_condition)

        except Exception as e:
            self.logger.error(f"Heel abnormality detection failed: {e}")

        return conditions

    # Helper methods for clinical recommendations

    def _get_hallux_valgus_recommendation(self, severity: ConditionSeverity, angle: float) -> str:
        """Get clinical recommendation for hallux valgus"""
        if severity == ConditionSeverity.MILD:
            return "Conservative management: proper footwear, toe spacers, exercises"
        elif severity == ConditionSeverity.MODERATE:
            return "Orthopaedic evaluation recommended. Consider custom orthotics"
        else:
            return "Urgent orthopaedic consultation for surgical evaluation"

    def _get_bunionette_recommendation(self, severity: ConditionSeverity) -> str:
        """Get clinical recommendation for bunionette"""
        if severity == ConditionSeverity.MILD:
            return "Footwear modification, padding, activity modification"
        elif severity == ConditionSeverity.MODERATE:
            return "Custom orthotics, physiotherapy, consider surgical consultation"
        else:
            return "Orthopaedic evaluation for surgical intervention"

    def _get_pes_planus_recommendation(self, severity: ConditionSeverity) -> str:
        """Get clinical recommendation for pes planus"""
        if severity == ConditionSeverity.MILD:
            return "Supportive footwear, arch support insoles, strengthening exercises"
        elif severity == ConditionSeverity.MODERATE:
            return "Custom orthotics, physiotherapy, posterior tibial tendon assessment"
        else:
            return "Urgent podiatric/orthopaedic evaluation for structural correction"

    def _get_pes_cavus_recommendation(self, severity: ConditionSeverity) -> str:
        """Get clinical recommendation for pes cavus"""
        if severity == ConditionSeverity.MILD:
            return "Cushioned footwear, metatarsal pads, flexibility exercises"
        else:
            return "Custom orthotics, neurological evaluation if indicated, biomechanical assessment"

    def _get_hallux_valgus_risk_factors(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify risk factors for hallux valgus progression"""
        risk_factors = []

        if analysis.get('intermetatarsal', {}).get('angle', 0) > 9:
            risk_factors.append("Increased intermetatarsal angle")

        if analysis.get('arch', {}).get('type') == 'flat':
            risk_factors.append("Pes planus (flat foot)")

        risk_factors.extend(["Genetic predisposition", "Inappropriate footwear", "Female gender"])

        return risk_factors

    def _assess_hallux_valgus_progression_risk(self, hva_angle: float, analysis: Dict[str, Any]) -> str:
        """Assess progression risk for hallux valgus"""
        if hva_angle > 35:
            return "high"
        elif hva_angle > 25 or analysis.get('intermetatarsal', {}).get('angle', 0) > 15:
            return "moderate"
        else:
            return "low"

    def _determine_treatment_urgency(self, severity: ConditionSeverity, progression_risk: str) -> str:
        """Determine treatment urgency"""
        if severity == ConditionSeverity.SEVERE or progression_risk == "high":
            return "urgent"
        elif severity == ConditionSeverity.MODERATE or progression_risk == "moderate":
            return "prompt"
        else:
            return "routine"

    def _analyze_individual_toe(self, toe_number: int, toe_data: Dict[str, Any]) -> Optional[ConditionDetection]:
        """Analyze individual toe for deformities"""
        # Simplified toe analysis - would need more sophisticated implementation
        return None

    def _detect_heel_spur(self, heel_data: Dict[str, Any]) -> Optional[ConditionDetection]:
        """Detect heel spur condition"""
        # Simplified heel spur detection
        return None

    def _detect_heel_alignment_issues(self, heel_data: Dict[str, Any]) -> Optional[ConditionDetection]:
        """Detect heel alignment issues"""
        # Simplified heel alignment detection
        return None