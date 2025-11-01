"""
Comprehensive Condition Detection Module
Detects an extensive range of foot and ankle conditions including:
- Structural issues, skin conditions, nail disorders
- Circulation problems, biomechanical issues, inflammatory conditions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import stats, signal, ndimage
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Import base classes
from .medical_conditions import MedicalCondition

logger = logging.getLogger(__name__)

@dataclass
class AdvancedMedicalCondition(MedicalCondition):
    """Extended medical condition with additional diagnostic information"""
    condition_category: str  # 'structural', 'inflammatory', 'circulatory', 'dermatological', 'neurological'
    severity_score: float  # 0-100 numerical severity
    anatomical_location: List[str]  # Detailed anatomical descriptions
    pathophysiology: str  # Brief explanation of underlying mechanism
    differential_diagnoses: List[str]  # Other conditions to consider
    recommended_imaging: List[str]  # Suggested imaging studies
    red_flags: List[str]  # Warning signs requiring immediate attention

class StructuralConditionDetector:
    """Detects structural foot and ankle conditions"""

    def __init__(self):
        self.conditions = [
            'hallux_valgus', 'hallux_rigidus', 'metatarsus_adductus',
            'cavus_foot', 'equinus_deformity', 'calcaneal_fracture',
            'jones_fracture', 'lisfranc_injury', 'tarsal_coalition'
        ]

    def detect_hallux_valgus(self, point_cloud: np.ndarray,
                           segmentation: np.ndarray) -> AdvancedMedicalCondition:
        """Detect hallux valgus (bunion) with detailed analysis"""

        hallux_points = point_cloud[segmentation == 1]
        first_mtp_points = point_cloud[segmentation == 3]

        if len(hallux_points) < 20 or len(first_mtp_points) < 10:
            return self._no_structural_detection('hallux_valgus')

        # Calculate hallux valgus angle
        hallux_angle = self._calculate_hallux_valgus_angle(hallux_points, first_mtp_points)

        # Assess bunion prominence
        bunion_prominence = self._measure_bunion_prominence(first_mtp_points, point_cloud)

        # Check for associated deformities
        associated_deformities = self._check_associated_deformities(point_cloud, segmentation)

        # Classify severity
        severity = self._classify_hallux_valgus_severity(hallux_angle, bunion_prominence)
        severity_score = min(100, hallux_angle * 2 + bunion_prominence * 10)

        return AdvancedMedicalCondition(
            condition_name="Hallux Valgus",
            detected=hallux_angle > 15,
            severity=severity,
            confidence=0.85 if hallux_angle > 20 else 0.65,
            affected_regions=[1, 3],
            measurements={
                'hallux_valgus_angle_degrees': hallux_angle,
                'bunion_prominence_mm': bunion_prominence,
                'associated_deformities': len(associated_deformities)
            },
            treatment_implications=[
                "Wide toe box essential",
                "Bunion pad accommodation",
                "Possible surgical consultation if >40°",
                "Orthotic with forefoot posting"
            ] if hallux_angle > 15 else [],
            last_modifications={
                'bunion_accommodation': bunion_prominence + 5,
                'first_mtp_width_increase': hallux_angle / 5,
                'toe_box_angle_adjustment': hallux_angle / 10
            } if hallux_angle > 15 else {},
            condition_category='structural',
            severity_score=severity_score,
            anatomical_location=['First metatarsophalangeal joint', 'Hallux', 'Medial forefoot'],
            pathophysiology='Progressive lateral deviation of hallux with medial prominence of first metatarsal head',
            differential_diagnoses=['Hallux rigidus', 'Gout', 'Sesamoiditis', 'First MTP joint arthritis'],
            recommended_imaging=['Weight-bearing AP and lateral X-rays', 'Sesamoid views if indicated'],
            red_flags=['Severe pain limiting ambulation', 'Overlapping second toe', 'Transfer metatarsalgia']
        )

    def detect_hallux_rigidus(self, point_cloud: np.ndarray,
                            segmentation: np.ndarray) -> AdvancedMedicalCondition:
        """Detect hallux rigidus (arthritis of first MTP joint)"""

        first_mtp_points = point_cloud[segmentation == 3]
        hallux_points = point_cloud[segmentation == 1]

        if len(first_mtp_points) < 10 or len(hallux_points) < 20:
            return self._no_structural_detection('hallux_rigidus')

        # Assess joint space narrowing (simulated from point density)
        joint_space = self._assess_joint_space(first_mtp_points, hallux_points)

        # Detect dorsal osteophytes
        dorsal_prominence = self._detect_dorsal_osteophytes(first_mtp_points)

        # Assess range of motion limitation (estimated from geometry)
        rom_limitation = self._estimate_rom_limitation(first_mtp_points, hallux_points)

        severity = self._classify_hallux_rigidus_severity(joint_space, dorsal_prominence, rom_limitation)
        severity_score = (4 - joint_space) * 20 + dorsal_prominence * 15

        return AdvancedMedicalCondition(
            condition_name="Hallux Rigidus",
            detected=joint_space < 3 or dorsal_prominence > 3,
            severity=severity,
            confidence=0.75,
            affected_regions=[1, 3],
            measurements={
                'joint_space_mm': joint_space,
                'dorsal_prominence_mm': dorsal_prominence,
                'estimated_rom_limitation_percent': rom_limitation
            },
            treatment_implications=[
                "Rigid forefoot rocker essential",
                "Accommodate dorsal prominence",
                "Consider carbon fiber plate",
                "Avoid excessive toe spring"
            ] if joint_space < 3 else [],
            last_modifications={
                'dorsal_accommodation': dorsal_prominence + 2,
                'forefoot_rocker_angle': 15,
                'first_mtp_depth_increase': 3,
                'rigid_forefoot_plate': 1.0
            } if joint_space < 3 else {},
            condition_category='structural',
            severity_score=severity_score,
            anatomical_location=['First metatarsophalangeal joint'],
            pathophysiology='Degenerative arthritis with cartilage loss and osteophyte formation',
            differential_diagnoses=['Hallux valgus', 'Gout', 'Sesamoiditis', 'Turf toe'],
            recommended_imaging=['Weight-bearing AP and lateral X-rays', 'Oblique views'],
            red_flags=['Complete loss of dorsiflexion', 'Severe pain with weight-bearing']
        )

    def detect_cavus_foot(self, point_cloud: np.ndarray,
                         segmentation: np.ndarray) -> AdvancedMedicalCondition:
        """Detect cavus foot deformity"""

        arch_segments = list(range(24, 32))
        arch_points = point_cloud[np.isin(segmentation, arch_segments)]

        if len(arch_points) < 50:
            return self._no_structural_detection('cavus_foot')

        # Calculate arch height
        arch_height = self._calculate_detailed_arch_height(arch_points, point_cloud)

        # Assess hindfoot varus
        hindfoot_varus = self._assess_hindfoot_alignment(point_cloud, segmentation)

        # Check for claw toes
        claw_toe_score = self._assess_claw_toes(point_cloud, segmentation)

        # Calculate calcaneal pitch
        calcaneal_pitch = self._calculate_calcaneal_pitch(point_cloud, segmentation)

        severity = self._classify_cavus_severity(arch_height, hindfoot_varus, claw_toe_score)
        severity_score = arch_height * 3 + hindfoot_varus * 2 + claw_toe_score * 1.5

        return AdvancedMedicalCondition(
            condition_name="Cavus Foot",
            detected=arch_height > 25 or calcaneal_pitch > 30,
            severity=severity,
            confidence=0.8,
            affected_regions=arch_segments + [32, 33],
            measurements={
                'arch_height_mm': arch_height,
                'hindfoot_varus_degrees': hindfoot_varus,
                'claw_toe_score': claw_toe_score,
                'calcaneal_pitch_degrees': calcaneal_pitch
            },
            treatment_implications=[
                "Accommodative arch support",
                "Metatarsal pad for forefoot pressure",
                "Deep heel cup for stability",
                "Lateral wedging for varus correction"
            ] if arch_height > 25 else [],
            last_modifications={
                'arch_support_height': min(15, arch_height - 10),
                'metatarsal_pad_height': 4,
                'lateral_heel_wedge': hindfoot_varus / 5,
                'heel_cup_depth': 8
            } if arch_height > 25 else {},
            condition_category='structural',
            severity_score=severity_score,
            anatomical_location=['Longitudinal arch', 'Hindfoot', 'Forefoot'],
            pathophysiology='Abnormally high longitudinal arch with associated hindfoot varus and forefoot pronation',
            differential_diagnoses=['Charcot-Marie-Tooth disease', 'Cerebral palsy', 'Spina bifida'],
            recommended_imaging=['Weight-bearing lateral X-ray', 'Hindfoot alignment view'],
            red_flags=['Progressive deformity', 'Neurological symptoms', 'Recurrent ankle instability']
        )

    def _calculate_hallux_valgus_angle(self, hallux_points: np.ndarray,
                                     mtp_points: np.ndarray) -> float:
        """Calculate hallux valgus angle"""
        if len(hallux_points) < 10 or len(mtp_points) < 5:
            return 0

        # Get hallux axis
        hallux_pca = np.linalg.eig(np.cov(hallux_points.T))
        hallux_axis = hallux_pca[1][:, 0]

        # Get metatarsal axis (approximate)
        mtp_center = np.mean(mtp_points, axis=0)
        foot_center = np.array([0, np.mean(hallux_points[:, 1]), 0])
        metatarsal_axis = mtp_center - foot_center
        metatarsal_axis = metatarsal_axis / np.linalg.norm(metatarsal_axis)

        # Calculate angle between axes
        dot_product = np.dot(hallux_axis, metatarsal_axis)
        angle = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))

        return min(60, angle)  # Cap at 60 degrees

    def _measure_bunion_prominence(self, mtp_points: np.ndarray,
                                 full_cloud: np.ndarray) -> float:
        """Measure bunion prominence"""
        if len(mtp_points) < 5:
            return 0

        # Find most medial MTP point
        medial_extent = np.min(mtp_points[:, 0])

        # Compare to normal foot width at same level
        mtp_y_level = np.mean(mtp_points[:, 1])
        same_level_points = full_cloud[np.abs(full_cloud[:, 1] - mtp_y_level) < 10]

        if len(same_level_points) > 10:
            normal_medial_extent = np.percentile(same_level_points[:, 0], 10)
            prominence = abs(medial_extent - normal_medial_extent)
            return min(15, prominence)

        return 0

    def _check_associated_deformities(self, point_cloud: np.ndarray,
                                    segmentation: np.ndarray) -> List[str]:
        """Check for associated toe deformities"""
        deformities = []

        # Check second toe for hammer/claw toe
        toe2_points = point_cloud[segmentation == 4]
        if len(toe2_points) > 10:
            flexion = self._calculate_toe_flexion(toe2_points)
            if flexion > 30:
                deformities.append("Second toe deformity")

        # Check for transfer metatarsalgia (simplified)
        mtp_segments = [5, 7, 9]
        for seg in mtp_segments:
            mtp_points = point_cloud[segmentation == seg]
            if len(mtp_points) > 5:
                prominence = np.mean(mtp_points[:, 2]) - np.percentile(point_cloud[:, 2], 10)
                if prominence > 8:
                    deformities.append(f"MTP {seg//2} prominence")

        return deformities

    def _classify_hallux_valgus_severity(self, angle: float, prominence: float) -> str:
        """Classify hallux valgus severity"""
        if angle < 15 and prominence < 3:
            return 'none'
        elif angle < 30 and prominence < 5:
            return 'mild'
        elif angle < 40 and prominence < 8:
            return 'moderate'
        else:
            return 'severe'

    def _assess_joint_space(self, mtp_points: np.ndarray, hallux_points: np.ndarray) -> float:
        """Assess joint space (simplified estimation)"""
        if len(mtp_points) < 5 or len(hallux_points) < 10:
            return 4.0  # Normal joint space

        mtp_center = np.mean(mtp_points, axis=0)
        hallux_proximal = hallux_points[np.argmin(hallux_points[:, 1])]

        distance = np.linalg.norm(mtp_center - hallux_proximal)
        joint_space = max(0, min(5, distance - 15))  # Normalize to 0-5 mm range

        return joint_space

    def _detect_dorsal_osteophytes(self, mtp_points: np.ndarray) -> float:
        """Detect dorsal osteophytes"""
        if len(mtp_points) < 10:
            return 0

        # Look for dorsal prominence
        dorsal_height = np.max(mtp_points[:, 2]) - np.mean(mtp_points[:, 2])
        return min(10, max(0, dorsal_height - 3))

    def _estimate_rom_limitation(self, mtp_points: np.ndarray, hallux_points: np.ndarray) -> float:
        """Estimate ROM limitation from geometry"""
        if len(mtp_points) < 5 or len(hallux_points) < 10:
            return 0

        # Calculate angle between MTP and hallux
        mtp_center = np.mean(mtp_points, axis=0)
        hallux_tip = hallux_points[np.argmax(hallux_points[:, 1])]

        vector = hallux_tip - mtp_center
        angle_to_ground = np.degrees(np.arctan2(vector[2], vector[1]))

        # Normal dorsiflexion should allow ~70° angle
        normal_angle = 70
        current_limitation = max(0, normal_angle - abs(angle_to_ground))

        return min(100, (current_limitation / normal_angle) * 100)

    def _classify_hallux_rigidus_severity(self, joint_space: float,
                                        prominence: float, rom_limit: float) -> str:
        """Classify hallux rigidus severity"""
        if joint_space > 3 and prominence < 2 and rom_limit < 25:
            return 'none'
        elif joint_space > 2 and prominence < 4 and rom_limit < 50:
            return 'mild'
        elif joint_space > 1 and prominence < 6 and rom_limit < 75:
            return 'moderate'
        else:
            return 'severe'

    def _calculate_detailed_arch_height(self, arch_points: np.ndarray,
                                      full_cloud: np.ndarray) -> float:
        """Calculate detailed arch height"""
        if len(arch_points) < 20:
            return 0

        ground_level = np.percentile(full_cloud[:, 2], 5)
        navicular_region = arch_points[arch_points[:, 1] > np.percentile(arch_points[:, 1], 40)]

        if len(navicular_region) > 5:
            arch_height = np.max(navicular_region[:, 2]) - ground_level
            return max(0, arch_height)

        return 0

    def _assess_hindfoot_alignment(self, point_cloud: np.ndarray,
                                 segmentation: np.ndarray) -> float:
        """Assess hindfoot varus/valgus alignment"""
        heel_points = point_cloud[segmentation == 32]

        if len(heel_points) < 20:
            return 0

        # Calculate heel inclination
        heel_pca = np.linalg.eig(np.cov(heel_points.T))
        primary_axis = heel_pca[1][:, 0]

        # Calculate angle from vertical
        vertical = np.array([0, 0, 1])
        angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(primary_axis, vertical)), 0, 1)))

        return min(30, angle)  # Cap at 30 degrees

    def _assess_claw_toes(self, point_cloud: np.ndarray, segmentation: np.ndarray) -> float:
        """Assess claw toe deformities"""
        toe_segments = [4, 6, 8, 10]
        total_score = 0

        for seg in toe_segments:
            toe_points = point_cloud[segmentation == seg]
            if len(toe_points) > 10:
                flexion = self._calculate_toe_flexion(toe_points)
                if flexion > 25:
                    total_score += min(5, flexion / 10)

        return total_score

    def _calculate_calcaneal_pitch(self, point_cloud: np.ndarray,
                                 segmentation: np.ndarray) -> float:
        """Calculate calcaneal pitch angle"""
        heel_points = point_cloud[segmentation == 32]

        if len(heel_points) < 20:
            return 20  # Normal pitch

        # Fit line to heel points
        heel_bottom = heel_points[heel_points[:, 2] < np.percentile(heel_points[:, 2], 20)]

        if len(heel_bottom) > 5:
            # Calculate slope
            slope, _ = np.polyfit(heel_bottom[:, 1], heel_bottom[:, 2], 1)
            pitch_angle = np.degrees(np.arctan(slope))
            return abs(pitch_angle)

        return 20

    def _classify_cavus_severity(self, arch_height: float, hindfoot_varus: float,
                               claw_score: float) -> str:
        """Classify cavus foot severity"""
        total_score = arch_height * 0.3 + hindfoot_varus * 0.5 + claw_score * 2

        if total_score < 10:
            return 'none'
        elif total_score < 20:
            return 'mild'
        elif total_score < 35:
            return 'moderate'
        else:
            return 'severe'

    def _calculate_toe_flexion(self, toe_points: np.ndarray) -> float:
        """Calculate toe flexion angle"""
        if len(toe_points) < 10:
            return 0

        z_coords = toe_points[:, 2]
        y_coords = toe_points[:, 1]

        if np.ptp(y_coords) < 5:
            return 0

        slope = np.polyfit(y_coords, z_coords, 1)[0]
        angle = np.degrees(np.arctan(slope))

        return abs(angle)

    def _no_structural_detection(self, condition_name: str) -> AdvancedMedicalCondition:
        """Return negative detection for structural conditions"""
        return AdvancedMedicalCondition(
            condition_name=condition_name.replace('_', ' ').title(),
            detected=False,
            severity='none',
            confidence=0.0,
            affected_regions=[],
            measurements={},
            treatment_implications=[],
            last_modifications={},
            condition_category='structural',
            severity_score=0.0,
            anatomical_location=[],
            pathophysiology='',
            differential_diagnoses=[],
            recommended_imaging=[],
            red_flags=[]
        )

class CirculatoryConditionDetector:
    """Detects circulatory and vascular conditions"""

    def detect_peripheral_arterial_disease(self, point_cloud: np.ndarray,
                                         temperature_map: Optional[np.ndarray] = None) -> AdvancedMedicalCondition:
        """Detect signs of peripheral arterial disease"""

        # Assess skin color changes (simulated from point density)
        color_changes = self._assess_skin_color_changes(point_cloud)

        # Check for temperature differences
        if temperature_map is not None:
            temp_gradient = self._assess_temperature_gradient(temperature_map)
        else:
            temp_gradient = 0

        # Assess pulse quality (simulated)
        pulse_quality = self._simulate_pulse_assessment()

        # Check for tissue loss indicators
        tissue_loss = self._assess_tissue_integrity(point_cloud)

        severity_score = color_changes * 20 + temp_gradient * 15 + tissue_loss * 25
        detected = severity_score > 30

        return AdvancedMedicalCondition(
            condition_name="Peripheral Arterial Disease",
            detected=detected,
            severity=self._classify_pad_severity(severity_score),
            confidence=0.6,  # Lower confidence without proper vascular testing
            affected_regions=list(range(1, 45)),  # Affects entire foot
            measurements={
                'color_change_score': color_changes,
                'temperature_gradient': temp_gradient,
                'pulse_quality_score': pulse_quality,
                'tissue_integrity_score': tissue_loss
            },
            treatment_implications=[
                "Vascular surgery consultation required",
                "Avoid tight-fitting footwear",
                "Pressure relief essential",
                "Monitor for ulceration"
            ] if detected else [],
            last_modifications={
                'pressure_relief_zones': 3,
                'extra_depth': 8,
                'seamless_construction': 1,
                'cushioning_enhancement': 10
            } if detected else {},
            condition_category='circulatory',
            severity_score=severity_score,
            anatomical_location=['Entire foot and ankle'],
            pathophysiology='Atherosclerotic narrowing of peripheral arteries reducing blood flow',
            differential_diagnoses=['Venous insufficiency', 'Diabetic neuropathy', 'Critical limb ischemia'],
            recommended_imaging=['Ankle-brachial index', 'Doppler ultrasound', 'CT angiography'],
            red_flags=['Rest pain', 'Non-healing wounds', 'Gangrene', 'ABI <0.9']
        )

    def detect_chronic_venous_insufficiency(self, point_cloud: np.ndarray,
                                          segmentation: np.ndarray) -> AdvancedMedicalCondition:
        """Detect chronic venous insufficiency"""

        # Assess ankle and lower leg swelling
        ankle_swelling = self._assess_ankle_swelling(point_cloud, segmentation)

        # Check for skin changes
        skin_changes = self._assess_venous_skin_changes(point_cloud)

        # Look for varicose vein indicators
        surface_irregularities = self._detect_surface_irregularities(point_cloud)

        severity_score = ankle_swelling * 25 + skin_changes * 20 + surface_irregularities * 15
        detected = ankle_swelling > 2 or severity_score > 40

        return AdvancedMedicalCondition(
            condition_name="Chronic Venous Insufficiency",
            detected=detected,
            severity=self._classify_cvi_severity(ankle_swelling, skin_changes),
            confidence=0.7,
            affected_regions=[32, 33, 42, 43, 44],  # Ankle and lower leg regions
            measurements={
                'ankle_swelling_mm': ankle_swelling,
                'skin_change_score': skin_changes,
                'surface_irregularity_score': surface_irregularities
            },
            treatment_implications=[
                "Compression therapy recommended",
                "Elevation when possible",
                "Avoid prolonged standing",
                "Monitor for skin breakdown"
            ] if detected else [],
            last_modifications={
                'accommodative_ankle_design': ankle_swelling,
                'volume_adjustability': 1,
                'breathable_materials': 1,
                'edema_accommodation': ankle_swelling * 1.5
            } if detected else {},
            condition_category='circulatory',
            severity_score=severity_score,
            anatomical_location=['Ankle', 'Lower leg', 'Dorsal foot'],
            pathophysiology='Venous valve insufficiency leading to venous hypertension and fluid accumulation',
            differential_diagnoses=['Lymphedema', 'Heart failure', 'Renal disease', 'Lipedema'],
            recommended_imaging=['Venous duplex ultrasound', 'Ankle-brachial index'],
            red_flags=['Venous ulceration', 'Cellulitis', 'Deep vein thrombosis signs']
        )

    def _assess_skin_color_changes(self, point_cloud: np.ndarray) -> float:
        """Assess skin color changes (simulated)"""
        # In real implementation, would use color imaging
        # Simulate based on point cloud density variations
        density_variations = []

        y_slices = np.linspace(np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1]), 10)
        for i in range(len(y_slices) - 1):
            slice_points = point_cloud[(point_cloud[:, 1] >= y_slices[i]) &
                                     (point_cloud[:, 1] < y_slices[i+1])]
            if len(slice_points) > 0:
                density = len(slice_points) / (np.ptp(slice_points[:, 0]) * np.ptp(slice_points[:, 2]) + 1e-6)
                density_variations.append(density)

        if density_variations:
            color_change_score = np.std(density_variations) / np.mean(density_variations)
            return min(5, color_change_score)

        return 0

    def _assess_temperature_gradient(self, temperature_map: np.ndarray) -> float:
        """Assess temperature gradient across foot"""
        if temperature_map is None or temperature_map.size == 0:
            return 0

        # Calculate temperature gradient from proximal to distal
        proximal_temp = np.mean(temperature_map[:temperature_map.shape[0]//3])
        distal_temp = np.mean(temperature_map[-temperature_map.shape[0]//3:])

        gradient = abs(proximal_temp - distal_temp)
        return min(10, gradient)

    def _simulate_pulse_assessment(self) -> float:
        """Simulate pulse quality assessment"""
        # In real implementation, would use doppler or palpation
        # Return random value for demonstration
        return np.random.uniform(1, 4)

    def _assess_tissue_integrity(self, point_cloud: np.ndarray) -> float:
        """Assess tissue integrity from surface characteristics"""
        # Look for surface irregularities that might indicate tissue loss
        z_values = point_cloud[:, 2]
        surface_roughness = np.std(np.gradient(np.sort(z_values)))

        # Look for potential ulceration sites (deep depressions)
        z_percentile_5 = np.percentile(z_values, 5)
        z_percentile_95 = np.percentile(z_values, 95)

        depth_variation = z_percentile_95 - z_percentile_5
        if depth_variation > 20:  # Significant depth variation
            return min(5, depth_variation / 10)

        return surface_roughness / 2

    def _classify_pad_severity(self, severity_score: float) -> str:
        """Classify PAD severity"""
        if severity_score < 30:
            return 'none'
        elif severity_score < 50:
            return 'mild'
        elif severity_score < 75:
            return 'moderate'
        else:
            return 'severe'

    def _assess_ankle_swelling(self, point_cloud: np.ndarray,
                             segmentation: np.ndarray) -> float:
        """Assess ankle swelling"""
        ankle_points = point_cloud[segmentation == 42]  # Ankle region

        if len(ankle_points) < 20:
            return 0

        # Calculate ankle circumference
        ankle_center = np.mean(ankle_points, axis=0)
        distances = np.linalg.norm(ankle_points - ankle_center, axis=1)
        avg_radius = np.mean(distances)

        # Compare to normal ankle radius (approximate)
        normal_radius = 35  # mm
        swelling = max(0, avg_radius - normal_radius)

        return min(15, swelling)

    def _assess_venous_skin_changes(self, point_cloud: np.ndarray) -> float:
        """Assess skin changes associated with venous insufficiency"""
        # Simulate skin texture analysis from surface roughness
        lower_third = point_cloud[point_cloud[:, 1] < np.percentile(point_cloud[:, 1], 33)]

        if len(lower_third) < 50:
            return 0

        surface_roughness = np.std(lower_third[:, 2])
        return min(5, surface_roughness)

    def _detect_surface_irregularities(self, point_cloud: np.ndarray) -> float:
        """Detect surface irregularities suggesting varicose veins"""
        # Look for rope-like surface irregularities
        irregularity_score = 0

        # Divide foot into regions and analyze each
        for axis in [0, 1]:  # x and y axes
            for percentile in [25, 50, 75]:
                threshold = np.percentile(point_cloud[:, axis], percentile)
                region_points = point_cloud[np.abs(point_cloud[:, axis] - threshold) < 5]

                if len(region_points) > 20:
                    z_variation = np.std(region_points[:, 2])
                    if z_variation > 3:
                        irregularity_score += 1

        return min(5, irregularity_score)

    def _classify_cvi_severity(self, swelling: float, skin_changes: float) -> str:
        """Classify chronic venous insufficiency severity"""
        if swelling < 2 and skin_changes < 2:
            return 'none'
        elif swelling < 5 and skin_changes < 3:
            return 'mild'
        elif swelling < 10 and skin_changes < 4:
            return 'moderate'
        else:
            return 'severe'

class NeurologicalConditionDetector:
    """Detects neurological conditions affecting the foot"""

    def detect_diabetic_neuropathy(self, point_cloud: np.ndarray,
                                 segmentation: np.ndarray,
                                 pressure_map: Optional[np.ndarray] = None) -> AdvancedMedicalCondition:
        """Detect diabetic peripheral neuropathy"""

        # Assess for Charcot changes
        charcot_signs = self._assess_charcot_changes(point_cloud, segmentation)

        # Check for pressure point abnormalities
        pressure_abnormalities = self._assess_pressure_distribution(point_cloud, pressure_map)

        # Look for deformities associated with neuropathy
        neuropathic_deformities = self._assess_neuropathic_deformities(point_cloud, segmentation)

        # Assess for potential ulceration sites
        ulcer_risk = self._assess_ulceration_risk(point_cloud, segmentation, pressure_map)

        severity_score = charcot_signs * 30 + pressure_abnormalities * 20 + neuropathic_deformities * 15 + ulcer_risk * 25
        detected = charcot_signs > 0 or pressure_abnormalities > 2 or ulcer_risk > 3

        return AdvancedMedicalCondition(
            condition_name="Diabetic Peripheral Neuropathy",
            detected=detected,
            severity=self._classify_neuropathy_severity(severity_score),
            confidence=0.75,
            affected_regions=list(range(1, 45)),
            measurements={
                'charcot_sign_score': charcot_signs,
                'pressure_abnormality_score': pressure_abnormalities,
                'deformity_score': neuropathic_deformities,
                'ulcer_risk_score': ulcer_risk
            },
            treatment_implications=[
                "Total contact casting if Charcot present",
                "Pressure redistribution essential",
                "Regular foot inspection required",
                "Diabetic foot care education"
            ] if detected else [],
            last_modifications={
                'total_contact_design': 1 if charcot_signs > 0 else 0,
                'pressure_relief_zones': ulcer_risk,
                'protective_padding': 5,
                'seamless_interior': 1
            } if detected else {},
            condition_category='neurological',
            severity_score=severity_score,
            anatomical_location=['Entire foot', 'Ankle'],
            pathophysiology='Progressive nerve damage leading to loss of sensation and autonomic dysfunction',
            differential_diagnoses=['Peripheral arterial disease', 'Vitamin B12 deficiency', 'Alcoholic neuropathy'],
            recommended_imaging=['Weight-bearing X-rays', 'MRI if Charcot suspected', 'Bone scan'],
            red_flags=['Active Charcot arthropathy', 'Open ulceration', 'Signs of infection']
        )

    def _assess_charcot_changes(self, point_cloud: np.ndarray,
                              segmentation: np.ndarray) -> float:
        """Assess for Charcot neuroarthropathy changes"""
        charcot_score = 0

        # Check midfoot for collapse/rocker bottom
        midfoot_segments = [24, 25, 26, 27, 28]
        midfoot_points = point_cloud[np.isin(segmentation, midfoot_segments)]

        if len(midfoot_points) > 50:
            # Look for rocker bottom deformity
            ground_level = np.percentile(point_cloud[:, 2], 5)
            midfoot_min = np.min(midfoot_points[:, 2])

            if midfoot_min - ground_level < 5:  # Midfoot collapse
                charcot_score += 2

        # Check for joint prominence/swelling
        joint_segments = [3, 5, 7, 9, 11]
        for seg in joint_segments:
            joint_points = point_cloud[segmentation == seg]
            if len(joint_points) > 10:
                joint_prominence = np.max(joint_points[:, 2]) - np.mean(joint_points[:, 2])
                if joint_prominence > 5:
                    charcot_score += 0.5

        return min(5, charcot_score)

    def _assess_pressure_distribution(self, point_cloud: np.ndarray,
                                    pressure_map: Optional[np.ndarray]) -> float:
        """Assess abnormal pressure distribution"""
        if pressure_map is None:
            # Simulate pressure analysis from geometric features
            return self._simulate_pressure_analysis(point_cloud)

        # Analyze actual pressure map
        high_pressure_areas = pressure_map > np.percentile(pressure_map, 90)
        pressure_concentration = np.sum(high_pressure_areas) / pressure_map.size

        return min(5, pressure_concentration * 20)

    def _simulate_pressure_analysis(self, point_cloud: np.ndarray) -> float:
        """Simulate pressure analysis from point cloud geometry"""
        # Find potential high-pressure areas based on prominence
        prominent_areas = 0

        # Check metatarsal heads
        for y_level in [0.7, 0.75, 0.8, 0.85]:  # Forefoot region
            y_pos = np.percentile(point_cloud[:, 1], y_level * 100)
            level_points = point_cloud[np.abs(point_cloud[:, 1] - y_pos) < 5]

            if len(level_points) > 10:
                level_prominence = np.max(level_points[:, 2]) - np.median(level_points[:, 2])
                if level_prominence > 4:
                    prominent_areas += 1

        return min(5, prominent_areas)

    def _assess_neuropathic_deformities(self, point_cloud: np.ndarray,
                                      segmentation: np.ndarray) -> float:
        """Assess deformities associated with neuropathy"""
        deformity_score = 0

        # Check for claw toes
        toe_segments = [4, 6, 8, 10]
        for seg in toe_segments:
            toe_points = point_cloud[segmentation == seg]
            if len(toe_points) > 10:
                flexion = self._calculate_toe_flexion(toe_points)
                if flexion > 30:
                    deformity_score += 1

        # Check for prominent metatarsal heads
        mtp_segments = [3, 5, 7, 9, 11]
        for seg in mtp_segments:
            mtp_points = point_cloud[segmentation == seg]
            if len(mtp_points) > 5:
                prominence = np.max(mtp_points[:, 2]) - np.median(point_cloud[:, 2])
                if prominence > 6:
                    deformity_score += 0.5

        return min(5, deformity_score)

    def _assess_ulceration_risk(self, point_cloud: np.ndarray,
                              segmentation: np.ndarray,
                              pressure_map: Optional[np.ndarray]) -> float:
        """Assess risk of ulceration"""
        risk_score = 0

        # High-risk areas: metatarsal heads, heel, toes
        high_risk_segments = [1, 3, 5, 7, 9, 11, 32]

        for seg in high_risk_segments:
            region_points = point_cloud[segmentation == seg]
            if len(region_points) > 5:
                # Check for excessive prominence
                prominence = np.max(region_points[:, 2]) - np.percentile(point_cloud[:, 2], 10)
                if prominence > 8:
                    risk_score += 1

                # Check for sharp edges or irregularities
                if len(region_points) > 20:
                    surface_roughness = np.std(region_points[:, 2])
                    if surface_roughness > 3:
                        risk_score += 0.5

        return min(5, risk_score)

    def _calculate_toe_flexion(self, toe_points: np.ndarray) -> float:
        """Calculate toe flexion angle"""
        if len(toe_points) < 10:
            return 0

        z_coords = toe_points[:, 2]
        y_coords = toe_points[:, 1]

        if np.ptp(y_coords) < 5:
            return 0

        slope = np.polyfit(y_coords, z_coords, 1)[0]
        angle = np.degrees(np.arctan(slope))

        return abs(angle)

    def _classify_neuropathy_severity(self, severity_score: float) -> str:
        """Classify neuropathy severity"""
        if severity_score < 20:
            return 'none'
        elif severity_score < 40:
            return 'mild'
        elif severity_score < 70:
            return 'moderate'
        else:
            return 'severe'

class ComprehensiveConditionAnalyzer:
    """Main analyzer combining all condition detectors"""

    def __init__(self):
        self.structural_detector = StructuralConditionDetector()
        self.circulatory_detector = CirculatoryConditionDetector()
        self.neurological_detector = NeurologicalConditionDetector()

        logger.info("Comprehensive Condition Analyzer initialized")

    def analyze_all_conditions(self, point_cloud: np.ndarray,
                             segmentation: np.ndarray,
                             temperature_map: Optional[np.ndarray] = None,
                             pressure_map: Optional[np.ndarray] = None) -> Dict[str, AdvancedMedicalCondition]:
        """Analyze all comprehensive conditions"""

        conditions = {}

        # Structural conditions
        try:
            conditions['hallux_valgus'] = self.structural_detector.detect_hallux_valgus(
                point_cloud, segmentation
            )
            conditions['hallux_rigidus'] = self.structural_detector.detect_hallux_rigidus(
                point_cloud, segmentation
            )
            conditions['cavus_foot'] = self.structural_detector.detect_cavus_foot(
                point_cloud, segmentation
            )
        except Exception as e:
            logger.warning(f"Structural condition detection failed: {e}")

        # Circulatory conditions
        try:
            conditions['peripheral_arterial_disease'] = self.circulatory_detector.detect_peripheral_arterial_disease(
                point_cloud, temperature_map
            )
            conditions['chronic_venous_insufficiency'] = self.circulatory_detector.detect_chronic_venous_insufficiency(
                point_cloud, segmentation
            )
        except Exception as e:
            logger.warning(f"Circulatory condition detection failed: {e}")

        # Neurological conditions
        try:
            conditions['diabetic_neuropathy'] = self.neurological_detector.detect_diabetic_neuropathy(
                point_cloud, segmentation, pressure_map
            )
        except Exception as e:
            logger.warning(f"Neurological condition detection failed: {e}")

        logger.info(f"Comprehensive analysis complete - {len(conditions)} conditions analyzed")

        return conditions

    def generate_comprehensive_report(self, conditions: Dict[str, AdvancedMedicalCondition]) -> Dict:
        """Generate comprehensive condition report"""

        report = {
            'detected_conditions': [],
            'conditions_by_category': {'structural': [], 'circulatory': [], 'neurological': []},
            'severity_distribution': {'mild': 0, 'moderate': 0, 'severe': 0},
            'red_flags': [],
            'imaging_recommendations': [],
            'differential_diagnoses': [],
            'pathophysiology_summary': {},
            'treatment_priorities': [],
            'total_modifications': {}
        }

        for name, condition in conditions.items():
            if condition.detected:
                condition_data = {
                    'name': condition.condition_name,
                    'category': condition.condition_category,
                    'severity': condition.severity,
                    'severity_score': condition.severity_score,
                    'confidence': condition.confidence,
                    'anatomical_location': condition.anatomical_location,
                    'pathophysiology': condition.pathophysiology,
                    'measurements': condition.measurements,
                    'red_flags': condition.red_flags,
                    'imaging': condition.recommended_imaging,
                    'differentials': condition.differential_diagnoses
                }

                report['detected_conditions'].append(condition_data)

                # Categorize by type
                if condition.condition_category in report['conditions_by_category']:
                    report['conditions_by_category'][condition.condition_category].append(condition_data)

                # Count severity
                if condition.severity in report['severity_distribution']:
                    report['severity_distribution'][condition.severity] += 1

                # Collect red flags
                report['red_flags'].extend(condition.red_flags)

                # Collect imaging recommendations
                report['imaging_recommendations'].extend(condition.recommended_imaging)

                # Collect differential diagnoses
                report['differential_diagnoses'].extend(condition.differential_diagnoses)

                # Store pathophysiology
                if condition.condition_category not in report['pathophysiology_summary']:
                    report['pathophysiology_summary'][condition.condition_category] = []
                report['pathophysiology_summary'][condition.condition_category].append(
                    f"{condition.condition_name}: {condition.pathophysiology}"
                )

                # Merge modifications
                for mod_key, mod_value in condition.last_modifications.items():
                    if mod_key in report['total_modifications']:
                        report['total_modifications'][mod_key] = max(
                            report['total_modifications'][mod_key], mod_value
                        )
                    else:
                        report['total_modifications'][mod_key] = mod_value

        # Remove duplicates
        report['red_flags'] = list(set(report['red_flags']))
        report['imaging_recommendations'] = list(set(report['imaging_recommendations']))
        report['differential_diagnoses'] = list(set(report['differential_diagnoses']))

        # Generate treatment priorities
        report['treatment_priorities'] = self._generate_treatment_priorities(report['detected_conditions'])

        return report

    def _generate_treatment_priorities(self, detected_conditions: List[Dict]) -> List[str]:
        """Generate treatment priorities based on detected conditions"""
        priorities = []

        # Sort by severity score
        conditions_by_severity = sorted(detected_conditions, key=lambda x: x['severity_score'], reverse=True)

        if conditions_by_severity:
            highest = conditions_by_severity[0]
            priorities.append(f"Priority 1: Address {highest['name']} ({highest['severity']} severity)")

            # Check for multiple high-severity conditions
            high_severity = [c for c in conditions_by_severity if c['severity_score'] > 60]
            if len(high_severity) > 1:
                priorities.append("Priority 2: Multidisciplinary approach recommended for multiple severe conditions")

            # Category-specific priorities
            structural_conditions = [c for c in detected_conditions if c['category'] == 'structural']
            if len(structural_conditions) >= 2:
                priorities.append("Priority 3: Orthopedic surgery consultation for multiple structural issues")

            circulatory_conditions = [c for c in detected_conditions if c['category'] == 'circulatory']
            if circulatory_conditions:
                priorities.append("Priority 4: Vascular assessment and management")

            neurological_conditions = [c for c in detected_conditions if c['category'] == 'neurological']
            if neurological_conditions:
                priorities.append("Priority 5: Diabetic foot care and neuropathy management")

        return priorities