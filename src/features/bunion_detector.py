"""
Bunion Detection Module
Identifies and quantifies bunions (hallux valgus) from foot point cloud
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class BunionAnalysis:
    """Results of bunion analysis"""
    has_bunion: bool
    severity: str  # 'none', 'mild', 'moderate', 'severe'
    hallux_valgus_angle: float  # Angle in degrees
    medial_prominence: float  # Protrusion in mm
    affected_volume: float  # Volume in mm³
    confidence: float  # Detection confidence 0-1
    affected_points: np.ndarray  # Indices of affected points

class BunionDetector:
    """Detects and analyzes bunions from segmented foot data"""

    # Severity thresholds based on hallux valgus angle
    SEVERITY_THRESHOLDS = {
        'mild': 15,      # 15-25 degrees
        'moderate': 25,   # 25-35 degrees
        'severe': 35      # >35 degrees
    }

    def __init__(self):
        self.ideal_foot_profile = None  # Will be loaded from database

    def detect(self, point_cloud: np.ndarray, 
              segmentation: np.ndarray,
              measurements: Optional[Dict] = None) -> BunionAnalysis:
        """
        Detect and analyze bunion from segmented foot point cloud

        Args:
            point_cloud: Nx3 array of points
            segmentation: N array of segment labels
            measurements: Optional foot measurements

        Returns:
            BunionAnalysis object with detection results
        """
        # Extract hallux (big toe) and metatarsal regions
        hallux_points = point_cloud[segmentation == 1]  # Hallux segment ID
        medial_ball_points = point_cloud[segmentation == 6]  # Medial ball segment ID
        metatarsal_points = point_cloud[segmentation == 8]  # First metatarsal segment ID

        if len(hallux_points) == 0 or len(metatarsal_points) == 0:
            logger.warning("Missing required segments for bunion detection")
            return BunionAnalysis(
                has_bunion=False,
                severity='none',
                hallux_valgus_angle=0,
                medial_prominence=0,
                affected_volume=0,
                confidence=0,
                affected_points=np.array([])
            )

        # Calculate hallux valgus angle
        hv_angle = self._calculate_hallux_valgus_angle(
            hallux_points, metatarsal_points
        )

        # Measure medial prominence
        prominence = self._calculate_medial_prominence(
            medial_ball_points, point_cloud
        )

        # Calculate affected volume
        volume = self._calculate_protrusion_volume(
            medial_ball_points, prominence
        )

        # Determine severity
        severity = self._classify_severity(hv_angle, prominence)

        # Identify affected points
        affected_indices = self._identify_affected_region(
            point_cloud, segmentation, prominence
        )

        # Calculate confidence score
        confidence = self._calculate_confidence(
            hv_angle, prominence, len(affected_indices)
        )

        has_bunion = severity != 'none'

        return BunionAnalysis(
            has_bunion=has_bunion,
            severity=severity,
            hallux_valgus_angle=hv_angle,
            medial_prominence=prominence,
            affected_volume=volume,
            confidence=confidence,
            affected_points=affected_indices
        )

    def _calculate_hallux_valgus_angle(self, hallux_points: np.ndarray,
                                       metatarsal_points: np.ndarray) -> float:
        """
        Calculate the hallux valgus angle

        Args:
            hallux_points: Points belonging to big toe
            metatarsal_points: Points belonging to first metatarsal

        Returns:
            Angle in degrees
        """
        # Fit lines through hallux and metatarsal using PCA
        hallux_center = hallux_points.mean(axis=0)
        metatarsal_center = metatarsal_points.mean(axis=0)

        # PCA for hallux direction
        hallux_centered = hallux_points - hallux_center
        hallux_cov = np.cov(hallux_centered.T)
        hallux_eigvals, hallux_eigvecs = np.linalg.eigh(hallux_cov)
        hallux_direction = hallux_eigvecs[:, np.argmax(hallux_eigvals)]

        # PCA for metatarsal direction
        meta_centered = metatarsal_points - metatarsal_center
        meta_cov = np.cov(meta_centered.T)
        meta_eigvals, meta_eigvecs = np.linalg.eigh(meta_cov)
        meta_direction = meta_eigvecs[:, np.argmax(meta_eigvals)]

        # Calculate angle between directions (project to XY plane)
        hallux_2d = hallux_direction[:2] / np.linalg.norm(hallux_direction[:2])
        meta_2d = meta_direction[:2] / np.linalg.norm(meta_direction[:2])

        cos_angle = np.dot(hallux_2d, meta_2d)
        angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
        angle_deg = np.degrees(angle_rad)

        # Check if hallux deviates medially (positive angle)
        cross_product = hallux_2d[0] * meta_2d[1] - hallux_2d[1] * meta_2d[0]
        if cross_product < 0:
            angle_deg = -angle_deg

        return abs(angle_deg)

    def _calculate_medial_prominence(self, medial_ball_points: np.ndarray,
                                    full_point_cloud: np.ndarray) -> float:
        """
        Calculate medial prominence (protrusion) in mm

        Args:
            medial_ball_points: Points in medial ball region
            full_point_cloud: Complete foot point cloud

        Returns:
            Prominence in mm
        """
        if len(medial_ball_points) == 0:
            return 0.0

        # Find the most medial points
        medial_indices = np.argsort(medial_ball_points[:, 0])[:10]  # X is medial-lateral
        most_medial_points = medial_ball_points[medial_indices]

        # Compare with ideal foot profile (simplified: use percentile approach)
        foot_width_at_ball = np.percentile(full_point_cloud[:, 0], 95) - \
                            np.percentile(full_point_cloud[:, 0], 5)

        # Expected medial boundary (simplified calculation)
        expected_medial = np.percentile(full_point_cloud[:, 0], 10)
        actual_medial = most_medial_points[:, 0].mean()

        prominence = max(0, expected_medial - actual_medial)

        return prominence

    def _calculate_protrusion_volume(self, affected_points: np.ndarray,
                                    prominence: float) -> float:
        """
        Calculate volume of bunion protrusion

        Args:
            affected_points: Points in affected region
            prominence: Medial prominence in mm

        Returns:
            Volume in mm³
        """
        if len(affected_points) == 0 or prominence == 0:
            return 0.0

        # Approximate volume using convex hull
        from scipy.spatial import ConvexHull

        try:
            hull = ConvexHull(affected_points)
            volume = hull.volume
        except:
            # Fallback: approximate as hemisphere
            radius = prominence / 2
            volume = (2/3) * np.pi * radius**3

        return volume

    def _classify_severity(self, hv_angle: float, prominence: float) -> str:
        """
        Classify bunion severity based on measurements

        Args:
            hv_angle: Hallux valgus angle in degrees
            prominence: Medial prominence in mm

        Returns:
            Severity classification
        """
        if hv_angle < 15 and prominence < 2:
            return 'none'
        elif hv_angle < self.SEVERITY_THRESHOLDS['mild'] or prominence < 5:
            return 'mild'
        elif hv_angle < self.SEVERITY_THRESHOLDS['moderate'] or prominence < 8:
            return 'moderate'
        else:
            return 'severe'

    def _identify_affected_region(self, point_cloud: np.ndarray,
                                 segmentation: np.ndarray,
                                 prominence: float) -> np.ndarray:
        """
        Identify points in the affected bunion region

        Args:
            point_cloud: Full point cloud
            segmentation: Segmentation labels
            prominence: Medial prominence

        Returns:
            Indices of affected points
        """
        # Get points from relevant segments
        bunion_segments = [6, 8, 19]  # Medial ball, first metatarsal, bunion area
        affected_mask = np.isin(segmentation, bunion_segments)

        # Further filter by medial position
        medial_threshold = np.percentile(point_cloud[:, 0], 20)
        medial_mask = point_cloud[:, 0] < medial_threshold

        final_mask = affected_mask & medial_mask
        affected_indices = np.where(final_mask)[0]

        return affected_indices

    def _calculate_confidence(self, hv_angle: float, prominence: float,
                            num_affected_points: int) -> float:
        """
        Calculate confidence score for bunion detection

        Args:
            hv_angle: Hallux valgus angle
            prominence: Medial prominence
            num_affected_points: Number of affected points

        Returns:
            Confidence score 0-1
        """
        # Base confidence on measurement reliability
        angle_confidence = min(1.0, hv_angle / 45)  # Max expected angle ~45°
        prominence_confidence = min(1.0, prominence / 10)  # Max expected ~10mm
        points_confidence = min(1.0, num_affected_points / 100)  # Expect 100+ points

        # Weighted average
        confidence = (angle_confidence * 0.5 + 
                     prominence_confidence * 0.3 + 
                     points_confidence * 0.2)

        return confidence

    def generate_correction_map(self, analysis: BunionAnalysis,
                               point_cloud: np.ndarray) -> np.ndarray:
        """
        Generate correction map for 3D printing

        Args:
            analysis: Bunion analysis results
            point_cloud: Full point cloud

        Returns:
            Correction values for each point (negative for relief pocket)
        """
        correction_map = np.zeros(len(point_cloud))

        if not analysis.has_bunion:
            return correction_map

        # Create relief pocket at bunion location
        for idx in analysis.affected_points:
            # Relief depth based on severity
            if analysis.severity == 'mild':
                depth = -2.0  # 2mm relief
            elif analysis.severity == 'moderate':
                depth = -3.5  # 3.5mm relief
            else:  # severe
                depth = -5.0  # 5mm relief

            correction_map[idx] = depth

        # Smooth transitions (simplified gaussian smoothing)
        from scipy.ndimage import gaussian_filter1d
        correction_map = gaussian_filter1d(correction_map, sigma=2)

        return correction_map