"""
Comprehensive Medical Condition Detection Module
Detects various foot conditions including collapsed arch, plantar fasciitis,
swollen feet, toe deformities, gout, flat feet, and more
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats, signal
from scipy.spatial import distance
import logging

logger = logging.getLogger(__name__)

@dataclass
class MedicalCondition:
    """Base class for medical condition analysis"""
    condition_name: str
    detected: bool
    severity: str  # 'none', 'mild', 'moderate', 'severe'
    confidence: float
    affected_regions: List[int]  # Segment indices
    measurements: Dict[str, float]
    treatment_implications: List[str]
    last_modifications: Dict[str, float]

class CollapsedArchDetector:
    """Detects collapsed arch (pes planus) condition"""

    def detect(self, point_cloud: np.ndarray, segmentation: np.ndarray,
              baseline_arch: Optional[np.ndarray] = None) -> MedicalCondition:
        """
        Detect collapsed arch condition

        Args:
            point_cloud: Foot point cloud
            segmentation: Segmentation labels
            baseline_arch: Healthy arch profile for comparison
        """
        # Extract arch regions (24-28 in enhanced segmentation)
        arch_segments = [24, 25, 26, 27, 28]
        arch_points = point_cloud[np.isin(segmentation, arch_segments)]

        if len(arch_points) == 0:
            return self._no_detection()

        # Calculate arch collapse metrics
        arch_height = self._calculate_arch_height(arch_points, point_cloud)
        arch_angle = self._calculate_arch_angle(arch_points)
        ground_contact_area = self._calculate_contact_area(arch_points)

        # Compare with baseline if available
        if baseline_arch is not None:
            deviation = self._compare_to_baseline(arch_points, baseline_arch)
        else:
            deviation = 0

        # Classify severity
        severity = self._classify_severity(arch_height, arch_angle, ground_contact_area)

        # Calculate confidence
        confidence = self._calculate_confidence(len(arch_points), arch_height)

        return MedicalCondition(
            condition_name="Collapsed Arch (Pes Planus)",
            detected=severity != 'none',
            severity=severity,
            confidence=confidence,
            affected_regions=arch_segments,
            measurements={
                'arch_height_mm': arch_height,
                'arch_angle_degrees': arch_angle,
                'ground_contact_percentage': ground_contact_area,
                'baseline_deviation_mm': deviation
            },
            treatment_implications=[
                "Requires maximum arch support",
                "Consider motion control features",
                "May need orthotic inserts",
                "Recommend stability shoe design"
            ] if severity != 'none' else [],
            last_modifications={
                'arch_support_height': 8.0 if severity == 'severe' else 5.0,
                'medial_posting': 4.0 if severity == 'severe' else 2.0,
                'heel_cup_depth': 2.0
            } if severity != 'none' else {}
        )

    def _calculate_arch_height(self, arch_points: np.ndarray, full_cloud: np.ndarray) -> float:
        """Calculate minimum arch height"""
        ground_level = np.percentile(full_cloud[:, 2], 2)
        arch_min = np.percentile(arch_points[:, 2], 10)
        return arch_min - ground_level

    def _calculate_arch_angle(self, arch_points: np.ndarray) -> float:
        """Calculate arch angle using PCA"""
        pca_result = np.linalg.eig(np.cov(arch_points.T))
        primary_axis = pca_result[1][:, 0]
        angle = np.degrees(np.arctan2(primary_axis[2], primary_axis[1]))
        return abs(angle)

    def _calculate_contact_area(self, arch_points: np.ndarray) -> float:
        """Calculate percentage of arch in contact with ground"""
        threshold = np.percentile(arch_points[:, 2], 20)
        contact_points = arch_points[arch_points[:, 2] < threshold]
        return (len(contact_points) / len(arch_points)) * 100

    def _compare_to_baseline(self, arch_points: np.ndarray, baseline: np.ndarray) -> float:
        """Compare to healthy baseline arch"""
        # Use Hausdorff distance for shape comparison
        return distance.directed_hausdorff(arch_points, baseline)[0]

    def _classify_severity(self, height: float, angle: float, contact: float) -> str:
        """Classify collapse severity"""
        if height > 10 and angle < 15 and contact < 30:
            return 'none'
        elif height > 7 and angle < 20 and contact < 40:
            return 'mild'
        elif height > 4 and angle < 25 and contact < 60:
            return 'moderate'
        else:
            return 'severe'

    def _calculate_confidence(self, num_points: int, height: float) -> float:
        """Calculate detection confidence"""
        points_confidence = min(1.0, num_points / 1000)
        height_confidence = 1.0 - min(1.0, abs(height - 15) / 15)
        return (points_confidence + height_confidence) / 2

    def _no_detection(self) -> MedicalCondition:
        """Return when no arch points found"""
        return MedicalCondition(
            condition_name="Collapsed Arch (Pes Planus)",
            detected=False,
            severity='none',
            confidence=0.0,
            affected_regions=[],
            measurements={},
            treatment_implications=[],
            last_modifications={}
        )

class PlantarFasciitisDetector:
    """Detects signs of plantar fasciitis"""

    def detect(self, point_cloud: np.ndarray, segmentation: np.ndarray,
              pressure_map: Optional[np.ndarray] = None) -> MedicalCondition:
        """Detect plantar fasciitis indicators"""

        # Extract plantar fascia regions (29-31)
        fascia_segments = [29, 30, 31, 32]  # Including heel
        fascia_points = point_cloud[np.isin(segmentation, fascia_segments)]

        if len(fascia_points) == 0:
            return self._no_detection()

        # Analyze fascia characteristics
        thickness = self._measure_fascia_thickness(fascia_points)
        heel_spur_present = self._detect_heel_spur(
            point_cloud[segmentation == 32]  # Heel region
        )
        inflammation_score = self._calculate_inflammation_pattern(fascia_points, pressure_map)

        # Classify severity
        severity = self._classify_severity(thickness, heel_spur_present, inflammation_score)

        return MedicalCondition(
            condition_name="Plantar Fasciitis",
            detected=severity != 'none',
            severity=severity,
            confidence=0.7 if severity != 'none' else 0.0,
            affected_regions=fascia_segments,
            measurements={
                'fascia_thickness_mm': thickness,
                'heel_spur_detected': heel_spur_present,
                'inflammation_score': inflammation_score
            },
            treatment_implications=[
                "Requires heel cushioning",
                "Need arch support to reduce fascia strain",
                "Consider heel lift",
                "Avoid thin, hard soles"
            ] if severity != 'none' else [],
            last_modifications={
                'heel_cushion_thickness': 8.0 if severity == 'severe' else 5.0,
                'arch_support_height': 4.0,
                'heel_lift': 3.0 if severity == 'severe' else 2.0,
                'forefoot_rocker': 2.0  # Reduce fascia stretch
            } if severity != 'none' else {}
        )

    def _measure_fascia_thickness(self, fascia_points: np.ndarray) -> float:
        """Estimate fascia thickness from point cloud density"""
        # Calculate point density as proxy for thickness
        if len(fascia_points) < 10:
            return 0

        # Use vertical spread as thickness indicator
        z_spread = np.percentile(fascia_points[:, 2], 90) - \
                  np.percentile(fascia_points[:, 2], 10)
        return z_spread

    def _detect_heel_spur(self, heel_points: np.ndarray) -> bool:
        """Detect potential heel spur from point irregularities"""
        if len(heel_points) < 50:
            return False

        # Look for protrusion in heel profile
        z_values = heel_points[:, 2]
        z_gradient = np.gradient(np.sort(z_values))

        # Heel spur creates sharp gradient change
        return np.max(np.abs(z_gradient)) > 2.0

    def _calculate_inflammation_pattern(self, fascia_points: np.ndarray,
                                       pressure_map: Optional[np.ndarray]) -> float:
        """Calculate inflammation likelihood from pressure patterns"""
        if pressure_map is None:
            # Use point distribution as proxy
            density_var = np.var([len(fascia_points[fascia_points[:, 1] < p])
                                 for p in np.percentile(fascia_points[:, 1], [25, 50, 75])])
            return min(1.0, density_var / 100)
        else:
            # Use actual pressure data
            return np.mean(pressure_map[pressure_map > np.percentile(pressure_map, 80)])

    def _classify_severity(self, thickness: float, heel_spur: bool, inflammation: float) -> str:
        """Classify plantar fasciitis severity"""
        if thickness < 4 and not heel_spur and inflammation < 0.3:
            return 'none'
        elif thickness < 5 and inflammation < 0.5:
            return 'mild'
        elif thickness < 6 or heel_spur or inflammation < 0.7:
            return 'moderate'
        else:
            return 'severe'

    def _no_detection(self) -> MedicalCondition:
        return MedicalCondition(
            condition_name="Plantar Fasciitis",
            detected=False,
            severity='none',
            confidence=0.0,
            affected_regions=[],
            measurements={},
            treatment_implications=[],
            last_modifications={}
        )

class SwollenFeetAnalyzer:
    """Analyzes foot swelling (edema)"""

    def analyze(self, point_cloud: np.ndarray, segmentation: np.ndarray,
               baseline_measurements: Optional[Dict] = None) -> MedicalCondition:
        """Detect and quantify foot swelling"""

        # Calculate volume and girth measurements
        volume = self._calculate_volume(point_cloud)
        ankle_girth = self._measure_girth(point_cloud, segmentation, 'ankle')
        instep_girth = self._measure_girth(point_cloud, segmentation, 'instep')
        forefoot_girth = self._measure_girth(point_cloud, segmentation, 'forefoot')

        # Compare to baseline if available
        if baseline_measurements:
            volume_increase = ((volume - baseline_measurements.get('volume', volume)) /
                             baseline_measurements.get('volume', volume)) * 100
            girth_increase = ((ankle_girth - baseline_measurements.get('ankle_girth', ankle_girth)) /
                            baseline_measurements.get('ankle_girth', ankle_girth)) * 100
        else:
            # Use statistical norms
            volume_increase = self._compare_to_norms(volume, 'volume')
            girth_increase = self._compare_to_norms(ankle_girth, 'ankle_girth')

        # Detect pitting edema pattern
        pitting_score = self._detect_pitting_pattern(point_cloud)

        # Classify severity
        severity = self._classify_severity(volume_increase, girth_increase, pitting_score)

        return MedicalCondition(
            condition_name="Foot Swelling (Edema)",
            detected=severity != 'none',
            severity=severity,
            confidence=0.8 if severity != 'none' else 0.0,
            affected_regions=list(range(32, 43)),  # Ankle and dorsal regions
            measurements={
                'volume_ml': volume,
                'volume_increase_percent': volume_increase,
                'ankle_girth_mm': ankle_girth,
                'instep_girth_mm': instep_girth,
                'forefoot_girth_mm': forefoot_girth,
                'girth_increase_percent': girth_increase,
                'pitting_score': pitting_score
            },
            treatment_implications=[
                "Requires adjustable closure system",
                "Need stretchable upper materials",
                "Extra volume in toe box and instep",
                "Avoid constrictive designs",
                "Consider compression features"
            ] if severity != 'none' else [],
            last_modifications={
                'volume_allowance': 15.0 if severity == 'severe' else 8.0,
                'instep_expansion': 10.0 if severity == 'severe' else 5.0,
                'ankle_circumference_increase': 8.0 if severity == 'severe' else 4.0,
                'toe_box_height_increase': 5.0
            } if severity != 'none' else {}
        )

    def _calculate_volume(self, point_cloud: np.ndarray) -> float:
        """Calculate foot volume using convex hull"""
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(point_cloud)
            return hull.volume / 1000  # Convert to ml
        except:
            return 0

    def _measure_girth(self, point_cloud: np.ndarray, segmentation: np.ndarray,
                      location: str) -> float:
        """Measure girth at specific location"""
        if location == 'ankle':
            y_pos = np.percentile(point_cloud[:, 1], 10)
        elif location == 'instep':
            y_pos = np.percentile(point_cloud[:, 1], 40)
        else:  # forefoot
            y_pos = np.percentile(point_cloud[:, 1], 70)

        # Get points near this y position
        slice_points = point_cloud[np.abs(point_cloud[:, 1] - y_pos) < 5]

        if len(slice_points) < 10:
            return 0

        # Calculate perimeter
        from scipy.spatial import ConvexHull
        try:
            hull_2d = ConvexHull(slice_points[:, [0, 2]])  # X-Z plane
            return hull_2d.area  # Perimeter in 2D
        except:
            return 0

    def _compare_to_norms(self, value: float, measurement_type: str) -> float:
        """Compare to population norms"""
        # Simplified norm comparison
        norms = {
            'volume': 350,  # ml
            'ankle_girth': 220  # mm
        }
        norm = norms.get(measurement_type, value)
        return ((value - norm) / norm) * 100

    def _detect_pitting_pattern(self, point_cloud: np.ndarray) -> float:
        """Detect pitting edema pattern from surface irregularities"""
        # Calculate surface roughness as indicator
        z_values = point_cloud[:, 2]
        roughness = np.std(np.gradient(np.sort(z_values)))
        return min(1.0, roughness / 2.0)

    def _classify_severity(self, volume_increase: float, girth_increase: float,
                          pitting: float) -> str:
        """Classify swelling severity"""
        if volume_increase < 5 and girth_increase < 5 and pitting < 0.2:
            return 'none'
        elif volume_increase < 10 and girth_increase < 10 and pitting < 0.4:
            return 'mild'
        elif volume_increase < 20 and girth_increase < 20 and pitting < 0.7:
            return 'moderate'
        else:
            return 'severe'

class ToeDeformityDetector:
    """Detects various toe deformities"""

    def detect_all(self, point_cloud: np.ndarray, segmentation: np.ndarray) -> List[MedicalCondition]:
        """Detect multiple toe deformity conditions"""
        conditions = []

        # Detect hammer toe
        conditions.append(self._detect_hammer_toe(point_cloud, segmentation))

        # Detect claw toe
        conditions.append(self._detect_claw_toe(point_cloud, segmentation))

        # Detect mallet toe
        conditions.append(self._detect_mallet_toe(point_cloud, segmentation))

        # Detect overlapping toes
        conditions.append(self._detect_overlapping_toes(point_cloud, segmentation))

        # Detect Morton's toe
        conditions.append(self._detect_mortons_toe(point_cloud, segmentation))

        return [c for c in conditions if c.detected]

    def _detect_hammer_toe(self, point_cloud: np.ndarray, segmentation: np.ndarray) -> MedicalCondition:
        """Detect hammer toe deformity"""

        # Analyze toe segments 4, 6, 8, 10 (lesser toes)
        affected_toes = []
        measurements = {}

        for toe_idx, seg_id in enumerate([4, 6, 8, 10], start=2):
            toe_points = point_cloud[segmentation == seg_id]
            if len(toe_points) < 20:
                continue

            # Check for abnormal flexion
            flexion_angle = self._calculate_toe_flexion(toe_points)
            if flexion_angle > 30:  # Abnormal flexion threshold
                affected_toes.append(seg_id)
                measurements[f'toe{toe_idx}_flexion'] = flexion_angle

        detected = len(affected_toes) > 0
        severity = self._classify_toe_severity(measurements)

        return MedicalCondition(
            condition_name="Hammer Toe",
            detected=detected,
            severity=severity,
            confidence=0.75 if detected else 0.0,
            affected_regions=affected_toes,
            measurements=measurements,
            treatment_implications=[
                "Requires extra toe box depth",
                "Need soft, stretchable upper",
                "Avoid narrow toe box",
                "Consider toe crest pad"
            ] if detected else [],
            last_modifications={
                'toe_box_height': 8.0 if severity == 'severe' else 5.0,
                'toe_box_width': 5.0,
                'upper_stretch_zones': 1.0  # Binary flag
            } if detected else {}
        )

    def _detect_claw_toe(self, point_cloud: np.ndarray, segmentation: np.ndarray) -> MedicalCondition:
        """Detect claw toe deformity"""

        affected_toes = []
        measurements = {}

        for toe_idx, seg_id in enumerate([4, 6, 8, 10], start=2):
            toe_points = point_cloud[segmentation == seg_id]
            mtp_points = point_cloud[segmentation == seg_id + 1]  # MTP joint

            if len(toe_points) < 20 or len(mtp_points) < 10:
                continue

            # Claw toe shows hyperextension at MTP and flexion at PIP/DIP
            mtp_angle = self._calculate_joint_angle(mtp_points, toe_points)
            toe_curl = self._calculate_toe_curl(toe_points)

            if mtp_angle > 20 and toe_curl > 40:
                affected_toes.append(seg_id)
                measurements[f'toe{toe_idx}_mtp_angle'] = mtp_angle
                measurements[f'toe{toe_idx}_curl'] = toe_curl

        detected = len(affected_toes) > 0
        severity = self._classify_toe_severity(measurements)

        return MedicalCondition(
            condition_name="Claw Toe",
            detected=detected,
            severity=severity,
            confidence=0.7 if detected else 0.0,
            affected_regions=affected_toes,
            measurements=measurements,
            treatment_implications=[
                "Requires significant toe box depth",
                "Need cushioned insole",
                "Metatarsal pad recommended",
                "Avoid high heels"
            ] if detected else [],
            last_modifications={
                'toe_box_height': 10.0 if severity == 'severe' else 6.0,
                'metatarsal_pad': 4.0,
                'toe_spring_reduction': -3.0  # Reduce toe spring
            } if detected else {}
        )

    def _detect_overlapping_toes(self, point_cloud: np.ndarray,
                                 segmentation: np.ndarray) -> MedicalCondition:
        """Detect overlapping toes"""

        # Check spacing between adjacent toes
        overlaps = []
        measurements = {}

        toe_segments = [(4, 6), (6, 8), (8, 10)]  # Adjacent toe pairs

        for idx, (seg1, seg2) in enumerate(toe_segments):
            toe1_points = point_cloud[segmentation == seg1]
            toe2_points = point_cloud[segmentation == seg2]

            if len(toe1_points) < 10 or len(toe2_points) < 10:
                continue

            # Calculate minimum distance
            min_dist = self._calculate_min_distance(toe1_points, toe2_points)

            if min_dist < 2.0:  # Overlapping threshold in mm
                overlaps.append((seg1, seg2))
                measurements[f'overlap_{idx}_distance'] = min_dist

        detected = len(overlaps) > 0

        return MedicalCondition(
            condition_name="Overlapping Toes",
            detected=detected,
            severity='moderate' if len(overlaps) > 1 else 'mild',
            confidence=0.8 if detected else 0.0,
            affected_regions=[seg for pair in overlaps for seg in pair],
            measurements=measurements,
            treatment_implications=[
                "Requires wide toe box",
                "Consider toe separators",
                "Avoid pointed toe shoes",
                "Need soft upper materials"
            ] if detected else [],
            last_modifications={
                'toe_box_width': 8.0,
                'toe_separator_spaces': 2.0,
                'upper_flexibility': 1.0
            } if detected else {}
        )

    def _detect_mortons_toe(self, point_cloud: np.ndarray,
                           segmentation: np.ndarray) -> MedicalCondition:
        """Detect Morton's toe (second toe longer than hallux)"""

        hallux_points = point_cloud[segmentation == 1]
        toe2_points = point_cloud[segmentation == 4]

        if len(hallux_points) < 20 or len(toe2_points) < 20:
            return self._no_toe_detection("Morton's Toe")

        # Compare toe lengths
        hallux_length = np.percentile(hallux_points[:, 1], 95) - \
                       np.percentile(hallux_points[:, 1], 5)
        toe2_length = np.percentile(toe2_points[:, 1], 95) - \
                     np.percentile(toe2_points[:, 1], 5)

        length_diff = toe2_length - hallux_length
        detected = length_diff > 3.0  # 3mm threshold

        return MedicalCondition(
            condition_name="Morton's Toe",
            detected=detected,
            severity='mild' if length_diff < 6 else 'moderate',
            confidence=0.9 if detected else 0.0,
            affected_regions=[1, 4] if detected else [],
            measurements={
                'hallux_length_mm': hallux_length,
                'second_toe_length_mm': toe2_length,
                'length_difference_mm': length_diff
            },
            treatment_implications=[
                "Adjust toe box shape",
                "Consider oblique toe design",
                "May need second toe cushioning"
            ] if detected else [],
            last_modifications={
                'toe_box_shape': 1.0,  # Flag for oblique shape
                'second_toe_allowance': length_diff
            } if detected else {}
        )

    def _detect_mallet_toe(self, point_cloud: np.ndarray,
                          segmentation: np.ndarray) -> MedicalCondition:
        """Detect mallet toe (DIP joint flexion)"""

        affected_toes = []
        measurements = {}

        for toe_idx, seg_id in enumerate([4, 6, 8, 10], start=2):
            toe_points = point_cloud[segmentation == seg_id]

            if len(toe_points) < 20:
                continue

            # Mallet toe affects distal joint
            tip_angle = self._calculate_toe_tip_angle(toe_points)

            if tip_angle > 25:
                affected_toes.append(seg_id)
                measurements[f'toe{toe_idx}_tip_angle'] = tip_angle

        detected = len(affected_toes) > 0
        severity = 'mild' if max(measurements.values(), default=0) < 35 else 'moderate'

        return MedicalCondition(
            condition_name="Mallet Toe",
            detected=detected,
            severity=severity,
            confidence=0.7 if detected else 0.0,
            affected_regions=affected_toes,
            measurements=measurements,
            treatment_implications=[
                "Need toe tip protection",
                "Soft toe box required",
                "Consider gel toe caps"
            ] if detected else [],
            last_modifications={
                'toe_tip_padding': 3.0,
                'toe_box_softness': 1.0
            } if detected else {}
        )

    def _calculate_toe_flexion(self, toe_points: np.ndarray) -> float:
        """Calculate toe flexion angle"""
        if len(toe_points) < 10:
            return 0

        # Fit line through toe points
        z_coords = toe_points[:, 2]
        y_coords = toe_points[:, 1]

        # Calculate angle from horizontal
        slope = np.polyfit(y_coords, z_coords, 1)[0]
        angle = np.degrees(np.arctan(slope))

        return abs(angle)

    def _calculate_joint_angle(self, joint_points: np.ndarray,
                              toe_points: np.ndarray) -> float:
        """Calculate angle at joint"""
        if len(joint_points) < 5 or len(toe_points) < 5:
            return 0

        joint_center = joint_points.mean(axis=0)
        toe_center = toe_points.mean(axis=0)

        vector = toe_center - joint_center
        angle = np.degrees(np.arctan2(vector[2], vector[1]))

        return abs(angle)

    def _calculate_toe_curl(self, toe_points: np.ndarray) -> float:
        """Calculate toe curl amount"""
        if len(toe_points) < 10:
            return 0

        # Calculate curvature
        y_coords = np.sort(toe_points[:, 1])
        z_coords = toe_points[np.argsort(toe_points[:, 1]), 2]

        # Fit polynomial and calculate curvature
        poly = np.polyfit(y_coords, z_coords, 2)
        curvature = abs(poly[0]) * 100  # Scale factor

        return curvature

    def _calculate_toe_tip_angle(self, toe_points: np.ndarray) -> float:
        """Calculate angle at toe tip"""
        if len(toe_points) < 10:
            return 0

        # Get distal portion (tip)
        tip_threshold = np.percentile(toe_points[:, 1], 80)
        tip_points = toe_points[toe_points[:, 1] > tip_threshold]

        if len(tip_points) < 5:
            return 0

        return self._calculate_toe_flexion(tip_points)

    def _calculate_min_distance(self, points1: np.ndarray,
                               points2: np.ndarray) -> float:
        """Calculate minimum distance between point sets"""
        from scipy.spatial import distance_matrix

        # Sample for efficiency
        sample_size = min(50, len(points1), len(points2))
        idx1 = np.random.choice(len(points1), sample_size, replace=False)
        idx2 = np.random.choice(len(points2), sample_size, replace=False)

        dist_matrix = distance_matrix(points1[idx1], points2[idx2])
        return np.min(dist_matrix)

    def _classify_toe_severity(self, measurements: Dict) -> str:
        """Classify toe deformity severity"""
        if not measurements:
            return 'none'

        max_angle = max(measurements.values())
        if max_angle < 30:
            return 'mild'
        elif max_angle < 45:
            return 'moderate'
        else:
            return 'severe'

    def _no_toe_detection(self, condition_name: str) -> MedicalCondition:
        return MedicalCondition(
            condition_name=condition_name,
            detected=False,
            severity='none',
            confidence=0.0,
            affected_regions=[],
            measurements={},
            treatment_implications=[],
            last_modifications={}
        )

class GoutDetector:
    """Detects signs of gout in feet"""

    def detect(self, point_cloud: np.ndarray, segmentation: np.ndarray,
              temperature_map: Optional[np.ndarray] = None) -> MedicalCondition:
        """Detect gout indicators"""

        # Gout commonly affects MTP joints, especially hallux
        mtp_segments = [3, 5, 7, 9, 11]  # All MTP joints
        affected_joints = []
        measurements = {}

        for seg_id in mtp_segments:
            joint_points = point_cloud[segmentation == seg_id]

            if len(joint_points) < 20:
                continue

            # Check for swelling/tophi
            swelling = self._detect_joint_swelling(joint_points, point_cloud)

            # Check for temperature elevation (if available)
            if temperature_map is not None:
                temp_elevation = self._check_temperature(joint_points, temperature_map)
            else:
                temp_elevation = 0

            if swelling > 3.0 or temp_elevation > 2.0:
                affected_joints.append(seg_id)
                measurements[f'joint_{seg_id}_swelling'] = swelling
                measurements[f'joint_{seg_id}_temp_elevation'] = temp_elevation

        # Check for tophi (uric acid deposits)
        tophi_detected = self._detect_tophi(point_cloud, segmentation)

        detected = len(affected_joints) > 0 or tophi_detected
        severity = self._classify_gout_severity(affected_joints, measurements, tophi_detected)

        return MedicalCondition(
            condition_name="Gout",
            detected=detected,
            severity=severity,
            confidence=0.65 if detected else 0.0,
            affected_regions=affected_joints,
            measurements=measurements,
            treatment_implications=[
                "Requires extra room at MTP joints",
                "Need soft, non-pressure materials",
                "Avoid tight fitting",
                "Consider easy on/off design",
                "Cushioned insole essential"
            ] if detected else [],
            last_modifications={
                'mtp_joint_room': 5.0 if severity == 'severe' else 3.0,
                'cushioning_thickness': 5.0,
                'pressure_relief_zones': 1.0,
                'adjustable_closure': 1.0
            } if detected else {}
        )

    def _detect_joint_swelling(self, joint_points: np.ndarray,
                              full_cloud: np.ndarray) -> float:
        """Detect joint swelling"""
        if len(joint_points) < 10:
            return 0

        # Calculate volume relative to expected
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(joint_points)
            volume = hull.volume

            # Compare to expected volume (simplified)
            expected_volume = len(joint_points) * 0.5  # Rough estimate
            swelling = (volume - expected_volume) / expected_volume * 10

            return max(0, swelling)
        except:
            return 0

    def _check_temperature(self, joint_points: np.ndarray,
                          temperature_map: np.ndarray) -> float:
        """Check for temperature elevation"""
        # In reality, would need thermal imaging
        # Here we simulate with density analysis
        return np.random.uniform(0, 3)  # Placeholder

    def _detect_tophi(self, point_cloud: np.ndarray,
                     segmentation: np.ndarray) -> bool:
        """Detect tophi (uric acid crystal deposits)"""
        # Look for irregular bumps near joints
        # Simplified detection based on surface roughness

        joint_segments = [3, 5, 7, 9, 11]
        for seg_id in joint_segments:
            joint_points = point_cloud[segmentation == seg_id]
            if len(joint_points) > 30:
                roughness = np.std(joint_points[:, 2])
                if roughness > 2.0:  # High surface variation
                    return True

        return False

    def _classify_gout_severity(self, affected_joints: List[int],
                               measurements: Dict, tophi: bool) -> str:
        """Classify gout severity"""
        if not affected_joints and not tophi:
            return 'none'
        elif len(affected_joints) == 1 and not tophi:
            return 'mild'
        elif len(affected_joints) <= 2 or tophi:
            return 'moderate'
        else:
            return 'severe'

class FlatFeetAnalyzer:
    """Comprehensive flat feet analysis"""

    def analyze(self, point_cloud: np.ndarray, segmentation: np.ndarray,
               weight_bearing: bool = True) -> MedicalCondition:
        """Perform comprehensive flat feet analysis"""

        # Extract all arch-related segments
        arch_segments = list(range(24, 32))  # All midfoot segments
        arch_points = point_cloud[np.isin(segmentation, arch_segments)]

        if len(arch_points) < 100:
            return self._no_detection()

        # Multiple flat feet indicators
        arch_index = self._calculate_arch_index(arch_points, point_cloud)
        navicular_drop = self._measure_navicular_drop(
            point_cloud[segmentation == 26]  # Navicular region
        )
        calcaneal_angle = self._measure_calcaneal_angle(
            point_cloud[segmentation == 33]  # Medial calcaneus
        )
        footprint_ratio = self._calculate_footprint_ratio(arch_points, point_cloud)

        # Classify flat feet type
        flat_type = self._classify_flat_feet_type(
            arch_index, navicular_drop, calcaneal_angle, weight_bearing
        )

        severity = self._determine_severity(
            arch_index, navicular_drop, calcaneal_angle, footprint_ratio
        )

        return MedicalCondition(
            condition_name=f"Flat Feet ({flat_type})",
            detected=severity != 'none',
            severity=severity,
            confidence=0.85 if severity != 'none' else 0.0,
            affected_regions=arch_segments,
            measurements={
                'arch_index': arch_index,
                'navicular_drop_mm': navicular_drop,
                'calcaneal_angle_degrees': calcaneal_angle,
                'footprint_ratio': footprint_ratio,
                'weight_bearing': weight_bearing
            },
            treatment_implications=[
                "Maximum arch support required",
                "Motion control features essential",
                "Medial posting recommended",
                "Firm heel counter needed",
                "Consider custom orthotics"
            ] if severity != 'none' else [],
            last_modifications={
                'arch_support_height': 10.0 if severity == 'severe' else 6.0,
                'medial_wedge': 5.0 if severity == 'severe' else 3.0,
                'heel_cup_depth': 5.0,
                'midfoot_stability_plate': 1.0,
                'pronation_control': 1.0
            } if severity != 'none' else {}
        )

    def _calculate_arch_index(self, arch_points: np.ndarray,
                             full_cloud: np.ndarray) -> float:
        """Calculate Cavanagh and Rodgers arch index"""
        # Divide foot into thirds and calculate contact area ratio
        foot_length = np.ptp(full_cloud[:, 1])

        # Get contact points (lowest 10%)
        contact_threshold = np.percentile(full_cloud[:, 2], 10)
        contact_points = arch_points[arch_points[:, 2] < contact_threshold]

        if len(contact_points) == 0:
            return 0

        # Calculate midfoot contact ratio
        midfoot_ratio = len(contact_points) / len(arch_points)

        return midfoot_ratio

    def _measure_navicular_drop(self, navicular_points: np.ndarray) -> float:
        """Measure navicular drop (key flat feet indicator)"""
        if len(navicular_points) < 10:
            return 0

        # Navicular height from ground
        ground_level = np.percentile(navicular_points[:, 2], 5)
        navicular_height = np.mean(navicular_points[:, 2]) - ground_level

        # Normal navicular height is ~15-20mm
        drop = max(0, 18 - navicular_height)

        return drop

    def _measure_calcaneal_angle(self, calcaneus_points: np.ndarray) -> float:
        """Measure calcaneal eversion angle"""
        if len(calcaneus_points) < 20:
            return 0

        # Fit plane to calcaneus points
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca.fit(calcaneus_points)

        # Get angle of primary axis
        primary_axis = pca.components_[0]
        angle = np.degrees(np.arctan2(primary_axis[0], primary_axis[2]))

        return abs(angle)

    def _calculate_footprint_ratio(self, arch_points: np.ndarray,
                                  full_cloud: np.ndarray) -> float:
        """Calculate footprint contact area ratio"""
        # Simulate footprint analysis
        contact_threshold = np.percentile(full_cloud[:, 2], 15)

        # Divide foot into regions
        foot_length = np.ptp(full_cloud[:, 1])
        hindfoot = full_cloud[full_cloud[:, 1] < np.percentile(full_cloud[:, 1], 33)]
        midfoot = full_cloud[(full_cloud[:, 1] >= np.percentile(full_cloud[:, 1], 33)) &
                           (full_cloud[:, 1] < np.percentile(full_cloud[:, 1], 66))]
        forefoot = full_cloud[full_cloud[:, 1] >= np.percentile(full_cloud[:, 1], 66)]

        # Calculate contact ratios
        midfoot_contact = len(midfoot[midfoot[:, 2] < contact_threshold]) / len(midfoot)

        return midfoot_contact

    def _classify_flat_feet_type(self, arch_index: float, navicular_drop: float,
                                calcaneal_angle: float, weight_bearing: bool) -> str:
        """Classify type of flat feet"""
        if not weight_bearing and arch_index > 0.3:
            return "Flexible Flat Feet"
        elif weight_bearing and arch_index > 0.4:
            return "Rigid Flat Feet"
        elif navicular_drop > 10:
            return "Adult Acquired Flat Feet"
        elif calcaneal_angle > 10:
            return "Posterior Tibial Tendon Dysfunction"
        else:
            return "Pes Planus"

    def _determine_severity(self, arch_index: float, navicular_drop: float,
                          calcaneal_angle: float, footprint_ratio: float) -> str:
        """Determine flat feet severity"""
        if arch_index < 0.21 and navicular_drop < 6:
            return 'none'
        elif arch_index < 0.26 and navicular_drop < 10:
            return 'mild'
        elif arch_index < 0.31 or navicular_drop < 15:
            return 'moderate'
        else:
            return 'severe'

    def _no_detection(self) -> MedicalCondition:
        return MedicalCondition(
            condition_name="Flat Feet",
            detected=False,
            severity='none',
            confidence=0.0,
            affected_regions=[],
            measurements={},
            treatment_implications=[],
            last_modifications={}
        )

class ComprehensiveMedicalAnalyzer:
    """Main analyzer that runs all medical condition detections"""

    def __init__(self):
        self.collapsed_arch = CollapsedArchDetector()
        self.plantar_fasciitis = PlantarFasciitisDetector()
        self.swollen_feet = SwollenFeetAnalyzer()
        self.toe_deformities = ToeDeformityDetector()
        self.gout = GoutDetector()
        self.flat_feet = FlatFeetAnalyzer()

    def analyze_foot(self, point_cloud: np.ndarray, segmentation: np.ndarray,
                    baseline_data: Optional[Dict] = None) -> Dict[str, MedicalCondition]:
        """Run comprehensive medical analysis on foot"""

        conditions = {}

        # Run all detectors
        conditions['collapsed_arch'] = self.collapsed_arch.detect(
            point_cloud, segmentation, baseline_data.get('baseline_arch') if baseline_data else None
        )

        conditions['plantar_fasciitis'] = self.plantar_fasciitis.detect(
            point_cloud, segmentation
        )

        conditions['swollen_feet'] = self.swollen_feet.analyze(
            point_cloud, segmentation, baseline_data.get('measurements') if baseline_data else None
        )

        # Toe deformities returns multiple conditions
        toe_conditions = self.toe_deformities.detect_all(point_cloud, segmentation)
        for condition in toe_conditions:
            conditions[condition.condition_name.lower().replace(' ', '_')] = condition

        conditions['gout'] = self.gout.detect(point_cloud, segmentation)

        conditions['flat_feet'] = self.flat_feet.analyze(point_cloud, segmentation)

        return conditions

    def generate_medical_report(self, conditions: Dict[str, MedicalCondition]) -> Dict:
        """Generate comprehensive medical report"""

        report = {
            'detected_conditions': [],
            'total_modifications': {},
            'priority_treatments': [],
            'last_requirements': {},
            'medical_summary': ""
        }

        # Compile detected conditions
        for name, condition in conditions.items():
            if condition.detected:
                report['detected_conditions'].append({
                    'name': condition.condition_name,
                    'severity': condition.severity,
                    'confidence': condition.confidence,
                    'measurements': condition.measurements
                })

                # Merge modifications
                for mod_key, mod_value in condition.last_modifications.items():
                    if mod_key in report['total_modifications']:
                        # Take maximum modification needed
                        report['total_modifications'][mod_key] = max(
                            report['total_modifications'][mod_key], mod_value
                        )
                    else:
                        report['total_modifications'][mod_key] = mod_value

                # Collect treatment implications
                report['priority_treatments'].extend(condition.treatment_implications)

        # Remove duplicate treatments
        report['priority_treatments'] = list(set(report['priority_treatments']))

        # Generate summary
        if report['detected_conditions']:
            severity_order = {'severe': 3, 'moderate': 2, 'mild': 1}
            report['detected_conditions'].sort(
                key=lambda x: severity_order.get(x['severity'], 0), reverse=True
            )

            report['medical_summary'] = self._generate_summary(report['detected_conditions'])
        else:
            report['medical_summary'] = "No significant medical conditions detected. Foot appears healthy."

        return report

    def _generate_summary(self, conditions: List[Dict]) -> str:
        """Generate medical summary text"""

        if not conditions:
            return "Healthy foot profile detected."

        primary = conditions[0]
        summary = f"Primary condition: {primary['name']} ({primary['severity']} severity). "

        if len(conditions) > 1:
            summary += f"Additional conditions detected: "
            summary += ", ".join([c['name'] for c in conditions[1:4]])  # List up to 3 more

            if len(conditions) > 4:
                summary += f", and {len(conditions) - 4} others"

        summary += ". Comprehensive last modifications required for optimal fit and comfort."

        return summary