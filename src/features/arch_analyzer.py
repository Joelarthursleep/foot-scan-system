"""
Arch Analysis Module
Analyzes foot arch characteristics and recommends support levels
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ArchAnalysis:
    """Results of arch analysis"""
    arch_type: str  # 'low', 'normal', 'high'
    arch_height_index: float  # AHI value
    arch_height_mm: float
    support_level_needed: str  # 'minimal', 'moderate', 'maximum'
    pressure_points: np.ndarray  # Predicted pressure point locations
    confidence: float

class ArchAnalyzer:
    """Analyzes foot arch characteristics"""

    # AHI classification thresholds
    AHI_THRESHOLDS = {
        'low': 0.25,
        'normal_low': 0.3,
        'normal_high': 0.35,
        'high': 0.4
    }

    def analyze(self, point_cloud: np.ndarray, 
               segmentation: np.ndarray,
               measurements: Optional[Dict] = None) -> ArchAnalysis:
        """
        Analyze foot arch from segmented point cloud

        Args:
            point_cloud: Nx3 array of points
            segmentation: N array of segment labels
            measurements: Optional foot measurements

        Returns:
            ArchAnalysis object
        """
        # Extract arch regions
        medial_arch = point_cloud[segmentation == 10]
        lateral_arch = point_cloud[segmentation == 11]
        heel = point_cloud[segmentation == 13]

        if len(medial_arch) == 0:
            logger.warning("No medial arch points found")
            return self._default_analysis()

        # Calculate arch height
        arch_height = self._calculate_arch_height(medial_arch, heel, point_cloud)

        # Calculate AHI
        ahi = self._calculate_ahi(arch_height, point_cloud)

        # Classify arch type
        arch_type = self._classify_arch(ahi)

        # Determine support level
        support_level = self._determine_support_level(arch_type, ahi)

        # Identify pressure points
        pressure_points = self._identify_pressure_points(
            arch_type, medial_arch, point_cloud
        )

        # Calculate confidence
        confidence = self._calculate_confidence(len(medial_arch), ahi)

        return ArchAnalysis(
            arch_type=arch_type,
            arch_height_index=ahi,
            arch_height_mm=arch_height,
            support_level_needed=support_level,
            pressure_points=pressure_points,
            confidence=confidence
        )

    def _calculate_arch_height(self, medial_arch: np.ndarray, 
                              heel: np.ndarray,
                              full_cloud: np.ndarray) -> float:
        """Calculate arch height in mm"""
        # Find lowest point (plantar surface)
        plantar_z = np.percentile(full_cloud[:, 2], 5)

        # Find highest point in medial arch
        arch_peak_z = np.percentile(medial_arch[:, 2], 95)

        arch_height = arch_peak_z - plantar_z
        return arch_height

    def _calculate_ahi(self, arch_height: float, point_cloud: np.ndarray) -> float:
        """Calculate Arch Height Index"""
        # Calculate truncated foot length (heel to ball)
        heel_y = np.percentile(point_cloud[:, 1], 5)
        ball_y = np.percentile(point_cloud[:, 1], 70)
        truncated_length = ball_y - heel_y

        if truncated_length > 0:
            ahi = arch_height / truncated_length
        else:
            ahi = 0.3  # Default normal value

        return ahi

    def _classify_arch(self, ahi: float) -> str:
        """Classify arch type based on AHI"""
        if ahi < self.AHI_THRESHOLDS['low']:
            return 'low'
        elif ahi < self.AHI_THRESHOLDS['normal_high']:
            return 'normal'
        else:
            return 'high'

    def _determine_support_level(self, arch_type: str, ahi: float) -> str:
        """Determine required support level"""
        if arch_type == 'low':
            if ahi < 0.15:
                return 'maximum'
            else:
                return 'moderate'
        elif arch_type == 'high':
            if ahi > 0.45:
                return 'maximum'
            else:
                return 'moderate'
        else:
            return 'minimal'

    def _identify_pressure_points(self, arch_type: str, 
                                 medial_arch: np.ndarray,
                                 full_cloud: np.ndarray) -> np.ndarray:
        """Identify potential pressure points based on arch type"""
        pressure_points = []

        if arch_type == 'low':
            # Pressure on medial midfoot
            medial_midfoot = full_cloud[
                (full_cloud[:, 0] < np.percentile(full_cloud[:, 0], 30)) &
                (full_cloud[:, 1] > np.percentile(full_cloud[:, 1], 30)) &
                (full_cloud[:, 1] < np.percentile(full_cloud[:, 1], 60))
            ]
            if len(medial_midfoot) > 0:
                pressure_points.extend(medial_midfoot[::10])  # Sample

        elif arch_type == 'high':
            # Pressure on heel and forefoot
            heel_points = full_cloud[full_cloud[:, 1] < np.percentile(full_cloud[:, 1], 20)]
            forefoot_points = full_cloud[full_cloud[:, 1] > np.percentile(full_cloud[:, 1], 70)]
            
            if len(heel_points) > 0:
                pressure_points.extend(heel_points[::20])
            if len(forefoot_points) > 0:
                pressure_points.extend(forefoot_points[::20])

        return np.array(pressure_points) if pressure_points else np.array([])

    def _calculate_confidence(self, num_arch_points: int, ahi: float) -> float:
        """Calculate confidence score"""
        # More points = higher confidence
        points_conf = min(1.0, num_arch_points / 500)

        # AHI in normal range = higher confidence
        ahi_conf = 1.0 - abs(ahi - 0.3) / 0.3

        return (points_conf + ahi_conf) / 2

    def _default_analysis(self) -> ArchAnalysis:
        """Return default analysis when data is insufficient"""
        return ArchAnalysis(
            arch_type='normal',
            arch_height_index=0.3,
            arch_height_mm=15.0,
            support_level_needed='minimal',
            pressure_points=np.array([]),
            confidence=0.0
        )

    def generate_support_profile(self, analysis: ArchAnalysis, 
                                point_cloud: np.ndarray) -> np.ndarray:
        """
        Generate arch support profile for custom last

        Args:
            analysis: Arch analysis results
            point_cloud: Full point cloud

        Returns:
            Support thickness map for each point
        """
        support_map = np.zeros(len(point_cloud))

        if analysis.support_level_needed == 'minimal':
            return support_map

        # Find arch region
        arch_y_min = np.percentile(point_cloud[:, 1], 30)
        arch_y_max = np.percentile(point_cloud[:, 1], 60)
        arch_mask = (point_cloud[:, 1] > arch_y_min) & (point_cloud[:, 1] < arch_y_max)

        if analysis.arch_type == 'low':
            # Add support to raise arch
            medial_mask = point_cloud[:, 0] < np.percentile(point_cloud[:, 0], 40)
            support_region = arch_mask & medial_mask

            if analysis.support_level_needed == 'moderate':
                support_map[support_region] = 3.0  # 3mm support
            else:  # maximum
                support_map[support_region] = 5.0  # 5mm support

        elif analysis.arch_type == 'high':
            # Add cushioning under high arch
            support_map[arch_mask] = 2.0  # 2mm cushioning

        # Smooth transitions
        from scipy.ndimage import gaussian_filter1d
        support_map = gaussian_filter1d(support_map, sigma=3)

        return support_map