#!/usr/bin/env python3
"""
Evidence-Based Condition Detector
Uses medical research data to detect and classify foot conditions with evidence backing
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .medical_research_loader import get_research_loader


class EvidenceBasedConditionDetector:
    """Detect foot conditions using evidence-based criteria from medical research"""

    def __init__(self):
        """Initialize the evidence-based detector"""
        self.research_loader = get_research_loader()
        self.detection_rules = self._build_detection_rules()

    def _build_detection_rules(self) -> Dict:
        """
        Build detection rules from research data

        Maps measurable 3D scan features to conditions based on clinical evidence
        """
        rules = {}

        # Get all conditions from research
        conditions = self.research_loader.get_all_conditions()

        for condition in conditions:
            condition_name = condition['name']
            rules[condition_name] = {
                'icd10': condition['icd10_code'],
                'category': condition['category'],
                'evidence_strength': condition['evidence_strength'],
                'detection_criteria': self._get_detection_criteria(condition_name, condition['category']),
                'symptoms': condition['symptoms'],
                'treatments': condition['treatments']
            }

        return rules

    def _get_detection_criteria(self, condition_name: str, category: str) -> Dict:
        """
        Define measurable detection criteria for each condition based on category

        These are evidence-based thresholds that can be measured from 3D scans
        """
        condition_lower = condition_name.lower()

        # Structural conditions - measurable from 3D geometry
        if category == 'structural':
            if 'hallux valgus' in condition_lower or 'bunion' in condition_lower:
                return {
                    'hallux_angle': {'min': 15, 'max': 180},  # degrees
                    'first_metatarsal_angle': {'min': 9, 'max': 180},
                    'detection_method': 'angle_measurement'
                }
            elif 'pes planus' in condition_lower or 'flat foot' in condition_lower:
                return {
                    'arch_height': {'min': 0, 'max': 15},  # mm
                    'arch_index': {'min': 0, 'max': 0.26},
                    'detection_method': 'arch_measurement'
                }
            elif 'pes cavus' in condition_lower or 'high arch' in condition_lower:
                return {
                    'arch_height': {'min': 30, 'max': 100},  # mm
                    'arch_index': {'min': 0.35, 'max': 1.0},
                    'detection_method': 'arch_measurement'
                }
            elif 'hammer' in condition_lower or 'claw toe' in condition_lower:
                return {
                    'toe_angle': {'min': 20, 'max': 180},  # degrees
                    'detection_method': 'toe_deformity'
                }

        # Bone conditions - measurable from 3D structure
        elif category == 'bone':
            if 'spur' in condition_lower:
                return {
                    'heel_prominence': {'min': 3, 'max': 100},  # mm
                    'detection_method': 'heel_geometry'
                }
            elif 'stress fracture' in condition_lower:
                return {
                    'asymmetry': {'min': 0.1, 'max': 1.0},
                    'detection_method': 'symmetry_analysis'
                }

        # Soft tissue conditions - measurable from volume/pressure
        elif category == 'soft_tissue':
            if 'plantar fasciitis' in condition_lower:
                return {
                    'heel_width': {'min': 1.05, 'max': 2.0},  # ratio to normal
                    'arch_strain': {'min': 0.15, 'max': 1.0},
                    'detection_method': 'plantar_analysis'
                }

        # Tendon conditions
        elif category == 'tendon':
            if 'achilles' in condition_lower:
                return {
                    'heel_angle': {'min': 15, 'max': 180},
                    'detection_method': 'heel_alignment'
                }

        # Pain conditions - require pressure mapping
        elif category == 'pain':
            if 'metatarsalgia' in condition_lower:
                return {
                    'metatarsal_pressure': {'min': 1.2, 'max': 5.0},  # ratio
                    'detection_method': 'pressure_analysis'
                }

        # Neurological - require symptom correlation
        elif category == 'neurological':
            if 'morton' in condition_lower:
                return {
                    'metatarsal_space': {'min': 0, 'max': 8},  # mm
                    'detection_method': 'spacing_analysis'
                }

        # Default criteria for conditions without specific rules
        return {
            'requires_manual_assessment': True,
            'detection_method': 'clinical_examination'
        }

    def detect_conditions(self, measurements, patient_data: Dict = None) -> List[Dict]:
        """
        Detect conditions from 3D scan measurements using evidence-based criteria

        Args:
            measurements: STLMeasurements object or dictionary of foot measurements from 3D scan
            patient_data: Optional patient demographics (age, activity_level, etc.)

        Returns:
            List of detected conditions with confidence scores and evidence
        """
        # Convert STLMeasurements object to dictionary if needed
        if hasattr(measurements, '__dict__'):
            measurements_dict = self._convert_measurements_to_dict(measurements)
        else:
            measurements_dict = measurements

        detected_conditions = []

        for condition_name, rules in self.detection_rules.items():
            # Check if condition can be detected from measurements
            detection_result = self._check_condition(
                condition_name,
                rules,
                measurements_dict,
                patient_data
            )

            if detection_result:
                detected_conditions.append(detection_result)

        # Sort by confidence score
        detected_conditions.sort(key=lambda x: x['confidence'], reverse=True)

        return detected_conditions

    def _convert_measurements_to_dict(self, measurements) -> Dict:
        """
        Convert STLMeasurements object to dictionary format for detection

        Args:
            measurements: STLMeasurements dataclass or similar object

        Returns:
            Dictionary with standardized measurement keys
        """
        measurements_dict = {}

        # Extract available measurements
        if hasattr(measurements, 'foot_length'):
            measurements_dict['foot_length'] = measurements.foot_length
        if hasattr(measurements, 'foot_width'):
            measurements_dict['foot_width'] = measurements.foot_width
        if hasattr(measurements, 'foot_height'):
            measurements_dict['foot_height'] = measurements.foot_height
        if hasattr(measurements, 'volume'):
            measurements_dict['volume'] = measurements.volume
        if hasattr(measurements, 'arch_height'):
            measurements_dict['arch_height'] = measurements.arch_height
        if hasattr(measurements, 'heel_width'):
            measurements_dict['heel_width'] = measurements.heel_width
        if hasattr(measurements, 'ball_girth'):
            measurements_dict['ball_girth'] = measurements.ball_girth

        # Calculate derived measurements that CAN detect conditions from STL data

        # Arch index (arch height / foot length) - CRITICAL for flat feet / high arch detection
        if 'arch_height' in measurements_dict and 'foot_length' in measurements_dict:
            if measurements_dict['foot_length'] > 0:
                measurements_dict['arch_index'] = measurements_dict['arch_height'] / measurements_dict['foot_length']

                # Calculate arch strain based on arch index (lower = more strain)
                # Normal arch_index is 0.26-0.35
                if measurements_dict['arch_index'] < 0.26:
                    # Flat foot - high strain
                    measurements_dict['arch_strain'] = 0.26 - measurements_dict['arch_index']
                elif measurements_dict['arch_index'] > 0.35:
                    # High arch - strain from excess curvature
                    measurements_dict['arch_strain'] = (measurements_dict['arch_index'] - 0.35) * 0.5
                else:
                    measurements_dict['arch_strain'] = 0.05  # Normal range

        # Heel width ratio (compared to foot width) - for plantar fasciitis detection
        if 'heel_width' in measurements_dict and 'foot_width' in measurements_dict:
            if measurements_dict['foot_width'] > 0:
                measurements_dict['heel_width_ratio'] = measurements_dict['heel_width'] / measurements_dict['foot_width']

        # Width ratio (foot width / foot length) - for wide foot detection
        if 'foot_width' in measurements_dict and 'foot_length' in measurements_dict:
            if measurements_dict['foot_length'] > 0:
                measurements_dict['width_ratio'] = measurements_dict['foot_width'] / measurements_dict['foot_length']
                # Normal ratio is typically 0.38-0.42 for adult feet
                # >0.42 = wide foot, <0.38 = narrow foot

        # Volume ratio (compared to expected volume from dimensions)
        if 'volume' in measurements_dict and 'foot_length' in measurements_dict and 'foot_width' in measurements_dict:
            # Expected volume approximation (ellipsoid formula)
            expected_volume = (4/3) * 3.14159 * (measurements_dict['foot_length']/2) * \
                             (measurements_dict['foot_width']/2) * (measurements_dict.get('foot_height', 50)/2)
            if expected_volume > 0:
                measurements_dict['volume_ratio'] = measurements_dict['volume'] / expected_volume
                # >1.1 might indicate swelling/edema, <0.9 might indicate atrophy

        # Height to length ratio - for foot structure analysis
        if 'foot_height' in measurements_dict and 'foot_length' in measurements_dict:
            if measurements_dict['foot_length'] > 0:
                measurements_dict['height_ratio'] = measurements_dict['foot_height'] / measurements_dict['foot_length']

        # Ball girth ratio - for forefoot issues
        if 'ball_girth' in measurements_dict and 'foot_length' in measurements_dict:
            if measurements_dict['foot_length'] > 0:
                measurements_dict['ball_girth_ratio'] = measurements_dict['ball_girth'] / measurements_dict['foot_length']

        # DO NOT SET DEFAULTS for measurements we can't calculate
        # Only detect conditions based on what we can actually measure from STL
        # This ensures we don't get false negatives from "normal" defaults

        return measurements_dict

    def _check_condition(
        self,
        condition_name: str,
        rules: Dict,
        measurements: Dict,
        patient_data: Dict = None
    ) -> Optional[Dict]:
        """
        Check if a specific condition is present based on measurements

        Args:
            condition_name: Name of condition to check
            rules: Detection rules for this condition
            measurements: Foot measurements from 3D scan
            patient_data: Patient demographics

        Returns:
            Detection result dict if condition detected, None otherwise
        """
        criteria = rules['detection_criteria']
        detection_method = criteria.get('detection_method')

        confidence = 0.0
        detected = False
        evidence = []

        # Structural conditions - angle-based detection
        if detection_method == 'angle_measurement':
            # SKIP: We can't measure hallux_angle from STL alone
            # Requires additional camera data or manual assessment
            return None

        elif detection_method == 'arch_measurement':
            if 'arch_height' in measurements:
                arch_height = measurements['arch_height']
                arch_index = measurements.get('arch_index', 0.3)

                if 'arch_height' in criteria:
                    if criteria['arch_height']['min'] <= arch_height <= criteria['arch_height']['max']:
                        detected = True
                        if 'pes planus' in condition_name.lower() or 'flat foot' in condition_name.lower():
                            # Flat foot detection
                            confidence = min(0.92, 0.55 + (15 - arch_height) / 15)
                            evidence.append(f"Arch height: {arch_height:.1f}mm (normal: 15-30mm)")
                            if arch_index < 0.26:
                                evidence.append(f"Arch index: {arch_index:.3f} (normal: 0.26-0.35)")
                                confidence = min(0.95, confidence + 0.05)
                        else:  # high arch
                            confidence = min(0.92, 0.55 + (arch_height - 30) / 70)
                            evidence.append(f"Arch height: {arch_height:.1f}mm (normal: 15-30mm)")
                            if arch_index > 0.35:
                                evidence.append(f"Arch index: {arch_index:.3f} (normal: 0.26-0.35)")
                                confidence = min(0.95, confidence + 0.05)

        elif detection_method == 'toe_deformity':
            # SKIP: We can't measure individual toe angles from STL alone
            # Requires additional camera data or manual assessment
            return None

        elif detection_method == 'heel_geometry':
            # SKIP: We can't measure heel prominence from STL alone yet
            # Would require advanced mesh analysis or additional sensors
            return None

        elif detection_method == 'plantar_analysis':
            if 'arch_strain' in measurements:
                strain = measurements['arch_strain']
                if strain >= criteria['arch_strain']['min']:
                    detected = True
                    confidence = min(0.88, 0.52 + strain * 1.5)
                    evidence.append(f"Arch strain index: {strain:.2f} (normal: <0.15)")

                    # Additional evidence from heel width if available
                    if 'heel_width_ratio' in measurements:
                        heel_ratio = measurements['heel_width_ratio']
                        if heel_ratio > 1.05:
                            confidence = min(0.92, confidence + 0.08)
                            evidence.append(f"Heel width ratio: {heel_ratio:.2f} (elevated, suggests inflammation)")

        elif detection_method == 'symmetry_analysis':
            # SKIP: L/R asymmetry requires scanning BOTH feet
            # Current implementation only scans one foot at a time
            return None

        elif detection_method == 'pressure_analysis':
            # SKIP: We can't measure pressure distribution from STL alone
            # Requires pressure mat or sensor array
            return None

        elif detection_method == 'heel_alignment':
            # SKIP: We can't measure heel angle from STL alone yet
            # Requires additional camera views or manual assessment
            return None

        elif detection_method == 'spacing_analysis':
            # SKIP: We can't measure metatarsal spacing from STL alone yet
            # Would require advanced mesh analysis
            return None

        # Apply patient-specific factors
        if detected and patient_data:
            confidence = self._adjust_confidence_for_patient(
                confidence,
                condition_name,
                rules,
                patient_data
            )

        if detected:
            return {
                'condition': condition_name,
                'icd10_code': rules['icd10'],
                'category': rules['category'],
                'confidence': confidence,
                'evidence_strength': rules['evidence_strength'],
                'evidence': evidence,
                'symptoms': rules['symptoms'][:5],  # Top 5 symptoms
                'treatments': rules['treatments'][:5],  # Top 5 treatments
                'detection_method': detection_method
            }

        return None

    def _adjust_confidence_for_patient(
        self,
        base_confidence: float,
        condition_name: str,
        rules: Dict,
        patient_data: Dict
    ) -> float:
        """
        Adjust confidence based on patient risk factors

        Args:
            base_confidence: Base confidence from measurement
            condition_name: Name of the condition
            rules: Detection rules
            patient_data: Patient demographics and history

        Returns:
            Adjusted confidence score
        """
        confidence = base_confidence
        age = patient_data.get('age', 45)
        activity_level = patient_data.get('activity_level', 50)

        condition_lower = condition_name.lower()

        # Age-related adjustments
        if 'hallux valgus' in condition_lower or 'bunion' in condition_lower:
            if age > 50:
                confidence = min(0.98, confidence + 0.05)  # More common in older adults

        if 'plantar fasciitis' in condition_lower:
            if 40 <= age <= 60:
                confidence = min(0.98, confidence + 0.05)  # Peak age range

        if 'achilles' in condition_lower:
            if activity_level > 70:
                confidence = min(0.98, confidence + 0.08)  # More common in athletes

        if 'metatarsalgia' in condition_lower:
            if activity_level > 60:
                confidence = min(0.98, confidence + 0.05)  # Common in active individuals

        # Metabolic conditions
        if 'diabetic' in condition_lower:
            if patient_data.get('diabetes', False):
                confidence = min(0.98, confidence + 0.15)

        # Arthritis
        if 'arthritis' in condition_lower:
            if age > 60:
                confidence = min(0.98, confidence + 0.10)

        return confidence

    def get_condition_info(self, condition_name: str) -> Optional[Dict]:
        """Get detailed information about a condition"""
        return self.research_loader.get_condition_by_name(condition_name)

    def get_treatment_recommendations(self, condition_name: str) -> List[str]:
        """Get evidence-based treatment recommendations"""
        return self.research_loader.get_treatment_recommendations(condition_name)

    def get_research_statistics(self) -> Dict:
        """Get statistics about the research database"""
        return self.research_loader.get_statistics()
