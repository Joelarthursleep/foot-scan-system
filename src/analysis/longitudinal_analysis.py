"""
Longitudinal Analysis Module for Foot Scan System

Implements advanced longitudinal analysis features including:
- ICP + TPS alignment for scan registration
- Regional delta computation
- Progression trend detection
- Clinical progression assessment

Based on the Step-by-Step Implementation Guide requirements
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)

@dataclass
class LongitudinalChange:
    """Represents a longitudinal change in foot morphology"""
    region: str
    baseline_value: float
    followup_value: float
    absolute_change: float
    percentage_change: float
    rate_per_month: float
    clinical_significance: str
    trend_direction: str
    confidence_score: float

@dataclass
class ProgressionTrend:
    """Represents progression trend analysis"""
    condition: str
    baseline_severity: str
    followup_severity: str
    progression_type: str  # 'improving', 'stable', 'worsening'
    rate_of_change: float
    predicted_severity_6mo: str
    predicted_severity_12mo: str
    clinical_recommendation: str

class LongitudinalAnalyzer:
    """Advanced longitudinal analysis with ICP + TPS alignment"""

    def __init__(self):
        """Initialize longitudinal analyzer"""
        self.logger = logging.getLogger(__name__)

    def align_scans_icp_tps(self, baseline_vertices: np.ndarray,
                           followup_vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Align follow-up scan to baseline using ICP + TPS (Thin Plate Spline) registration

        Args:
            baseline_vertices: Baseline scan vertices (Nx3)
            followup_vertices: Follow-up scan vertices (Mx3)

        Returns:
            Tuple of (aligned_followup_vertices, alignment_matrix, alignment_quality_metrics)
        """
        try:
            # Step 1: Initial ICP alignment
            aligned_followup, icp_transform = self._perform_icp_alignment(
                baseline_vertices, followup_vertices
            )

            # Step 2: TPS refinement for non-rigid deformation
            tps_aligned_followup, tps_params = self._apply_tps_refinement(
                baseline_vertices, aligned_followup
            )

            # Step 3: Compute alignment quality metrics
            quality_metrics = self._compute_alignment_quality(
                baseline_vertices, tps_aligned_followup, icp_transform, tps_params
            )

            return tps_aligned_followup, icp_transform, quality_metrics

        except Exception as e:
            self.logger.error(f"Scan alignment failed: {e}")
            # Fallback to simple centroid alignment
            return self._fallback_centroid_alignment(baseline_vertices, followup_vertices)

    def _perform_icp_alignment(self, baseline: np.ndarray,
                              followup: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform Iterative Closest Point alignment"""
        try:
            # Simplified ICP implementation
            max_iterations = 50
            tolerance = 1e-6

            # Center both point clouds
            baseline_centroid = np.mean(baseline, axis=0)
            followup_centroid = np.mean(followup, axis=0)

            baseline_centered = baseline - baseline_centroid
            followup_centered = followup - followup_centroid

            # Initial transformation matrix
            transform = np.eye(4)
            current_followup = followup_centered.copy()

            for iteration in range(max_iterations):
                # Find closest points (simplified - use subset for performance)
                n_samples = min(1000, len(baseline_centered), len(current_followup))
                baseline_sample = baseline_centered[::len(baseline_centered)//n_samples][:n_samples]
                followup_sample = current_followup[::len(current_followup)//n_samples][:n_samples]

                # Compute transformation using SVD
                try:
                    # Cross-covariance matrix
                    H = followup_sample.T @ baseline_sample
                    U, S, Vt = np.linalg.svd(H)
                    R = Vt.T @ U.T

                    # Handle reflection case
                    if np.linalg.det(R) < 0:
                        Vt[-1, :] *= -1
                        R = Vt.T @ U.T

                    # Translation
                    t = baseline_centroid - R @ followup_centroid

                    # Apply transformation
                    current_followup = (R @ followup_centered.T).T + t

                    # Update transformation matrix
                    new_transform = np.eye(4)
                    new_transform[:3, :3] = R
                    new_transform[:3, 3] = t
                    transform = new_transform @ transform

                except np.linalg.LinAlgError:
                    break

            return current_followup, transform

        except Exception as e:
            self.logger.error(f"ICP alignment failed: {e}")
            return followup, np.eye(4)

    def _apply_tps_refinement(self, baseline: np.ndarray,
                             aligned_followup: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply Thin Plate Spline refinement for non-rigid deformation"""
        try:
            # Simplified TPS - select control points
            n_control_points = min(100, len(baseline) // 10)

            # Select control points from baseline (evenly distributed)
            control_indices = np.linspace(0, len(baseline)-1, n_control_points, dtype=int)
            control_points_baseline = baseline[control_indices]

            # Find corresponding points in aligned follow-up
            control_points_followup = self._find_closest_points(
                control_points_baseline, aligned_followup
            )

            # Compute displacement field at control points
            displacements = control_points_followup - control_points_baseline

            # Apply smoothing to displacement field
            smoothed_displacements = self._smooth_displacement_field(
                control_points_baseline, displacements
            )

            # Interpolate displacement field to all follow-up points
            interpolated_displacements = self._interpolate_displacements(
                control_points_baseline, smoothed_displacements, aligned_followup
            )

            # Apply TPS transformation
            tps_aligned = aligned_followup + interpolated_displacements

            tps_params = {
                'control_points': control_points_baseline,
                'displacements': smoothed_displacements,
                'n_control_points': n_control_points
            }

            return tps_aligned, tps_params

        except Exception as e:
            self.logger.error(f"TPS refinement failed: {e}")
            return aligned_followup, {}

    def _find_closest_points(self, query_points: np.ndarray,
                            target_points: np.ndarray) -> np.ndarray:
        """Find closest points in target to each query point"""
        closest_points = np.zeros_like(query_points)

        for i, query_point in enumerate(query_points):
            distances = np.linalg.norm(target_points - query_point, axis=1)
            closest_idx = np.argmin(distances)
            closest_points[i] = target_points[closest_idx]

        return closest_points

    def _smooth_displacement_field(self, points: np.ndarray,
                                  displacements: np.ndarray) -> np.ndarray:
        """Smooth displacement field using Gaussian weighting"""
        smoothed = np.zeros_like(displacements)
        sigma = np.mean(np.std(points, axis=0)) * 0.1  # Adaptive sigma

        for i in range(len(points)):
            distances = np.linalg.norm(points - points[i], axis=1)
            weights = np.exp(-distances**2 / (2 * sigma**2))
            weights /= np.sum(weights)

            smoothed[i] = np.sum(displacements * weights[:, np.newaxis], axis=0)

        return smoothed

    def _interpolate_displacements(self, control_points: np.ndarray,
                                  control_displacements: np.ndarray,
                                  target_points: np.ndarray) -> np.ndarray:
        """Interpolate displacement field to target points using RBF"""
        interpolated = np.zeros_like(target_points)
        sigma = np.mean(np.std(control_points, axis=0)) * 0.5

        for i, target_point in enumerate(target_points):
            distances = np.linalg.norm(control_points - target_point, axis=1)
            weights = np.exp(-distances**2 / (2 * sigma**2))
            weights /= np.sum(weights) if np.sum(weights) > 0 else 1

            interpolated[i] = np.sum(control_displacements * weights[:, np.newaxis], axis=0)

        return interpolated

    def _compute_alignment_quality(self, baseline: np.ndarray, aligned: np.ndarray,
                                  icp_transform: np.ndarray, tps_params: Dict[str, Any]) -> Dict[str, Any]:
        """Compute alignment quality metrics"""
        try:
            # Root Mean Square Error
            n_samples = min(1000, len(baseline), len(aligned))
            baseline_sample = baseline[::len(baseline)//n_samples][:n_samples]
            aligned_sample = aligned[::len(aligned)//n_samples][:n_samples]

            distances = []
            for point in aligned_sample:
                dist_to_baseline = np.linalg.norm(baseline_sample - point, axis=1)
                distances.append(np.min(dist_to_baseline))

            rmse = np.sqrt(np.mean(np.array(distances)**2))

            # Hausdorff distance approximation
            max_dist = np.max(distances)

            # Transformation determinant (measure of scaling)
            det = np.linalg.det(icp_transform[:3, :3])

            return {
                'rmse': float(rmse),
                'max_distance': float(max_dist),
                'transformation_determinant': float(det),
                'n_control_points': tps_params.get('n_control_points', 0),
                'alignment_quality': 'good' if rmse < 2.0 else 'fair' if rmse < 5.0 else 'poor'
            }

        except Exception as e:
            return {
                'rmse': float('inf'),
                'max_distance': float('inf'),
                'alignment_quality': 'failed',
                'error': str(e)
            }

    def _fallback_centroid_alignment(self, baseline: np.ndarray,
                                   followup: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Fallback alignment using centroid matching"""
        baseline_centroid = np.mean(baseline, axis=0)
        followup_centroid = np.mean(followup, axis=0)

        translation = baseline_centroid - followup_centroid
        aligned_followup = followup + translation

        transform = np.eye(4)
        transform[:3, 3] = translation

        quality = {
            'rmse': float('inf'),
            'alignment_quality': 'fallback',
            'method': 'centroid_only'
        }

        return aligned_followup, transform, quality

    def compute_regional_deltas(self, baseline_analysis: Dict[str, Any],
                               followup_analysis: Dict[str, Any],
                               time_months: float) -> Dict[str, LongitudinalChange]:
        """
        Compute regional changes between baseline and follow-up scans

        Args:
            baseline_analysis: Baseline foot structure analysis
            followup_analysis: Follow-up foot structure analysis
            time_months: Time elapsed in months

        Returns:
            Dictionary of regional changes
        """
        regional_changes = {}

        # Define regions to analyze
        regions_to_analyze = [
            ('arch', 'ahi', 'Arch Height Index'),
            ('arch', 'height', 'Arch Height'),
            ('hallux_valgus', 'angle' if 'angle' in followup_analysis.get('hallux_valgus', {}) else 'hva', 'Hallux Valgus Angle'),
            ('instep', 'height', 'Instep Height'),
            ('dimensions', 'length', 'Foot Length'),
            ('dimensions', 'width', 'Foot Width'),
        ]

        for region_key, metric_key, clinical_name in regions_to_analyze:
            try:
                baseline_val = self._get_nested_value(baseline_analysis, region_key, metric_key)
                followup_val = self._get_nested_value(followup_analysis, region_key, metric_key)

                if baseline_val is not None and followup_val is not None:
                    change = self._compute_longitudinal_change(
                        baseline_val, followup_val, time_months, clinical_name
                    )
                    regional_changes[f"{region_key}_{metric_key}"] = change

            except Exception as e:
                self.logger.error(f"Failed to compute change for {clinical_name}: {e}")

        return regional_changes

    def _get_nested_value(self, data: Dict[str, Any], *keys) -> Optional[float]:
        """Safely get nested dictionary value"""
        try:
            result = data
            for key in keys:
                result = result[key]
            return float(result) if result is not None else None
        except (KeyError, TypeError, ValueError):
            return None

    def _compute_longitudinal_change(self, baseline: float, followup: float,
                                   time_months: float, region_name: str) -> LongitudinalChange:
        """Compute longitudinal change metrics"""
        absolute_change = followup - baseline
        percentage_change = (absolute_change / baseline * 100) if baseline != 0 else 0
        rate_per_month = absolute_change / time_months if time_months > 0 else 0

        # Assess clinical significance
        significance = self._assess_clinical_significance(
            region_name, absolute_change, percentage_change
        )

        # Determine trend direction
        if abs(percentage_change) < 2:
            trend_direction = 'stable'
        elif percentage_change > 0:
            trend_direction = 'increasing'
        else:
            trend_direction = 'decreasing'

        # Confidence score based on magnitude of change
        confidence_score = min(1.0, abs(percentage_change) / 10)

        return LongitudinalChange(
            region=region_name,
            baseline_value=baseline,
            followup_value=followup,
            absolute_change=absolute_change,
            percentage_change=percentage_change,
            rate_per_month=rate_per_month,
            clinical_significance=significance,
            trend_direction=trend_direction,
            confidence_score=confidence_score
        )

    def _assess_clinical_significance(self, region_name: str,
                                    absolute_change: float,
                                    percentage_change: float) -> str:
        """Assess clinical significance of change"""
        # Clinical thresholds for different regions
        thresholds = {
            'Arch Height Index': {'significant': 3, 'major': 8},
            'Arch Height': {'significant': 2, 'major': 5},
            'Hallux Valgus Angle': {'significant': 5, 'major': 10},
            'Instep Height': {'significant': 2, 'major': 5},
            'Foot Length': {'significant': 2, 'major': 5},
            'Foot Width': {'significant': 1, 'major': 3},
        }

        threshold = thresholds.get(region_name, {'significant': 5, 'major': 15})
        abs_change = abs(absolute_change)

        if abs_change >= threshold['major']:
            return 'major'
        elif abs_change >= threshold['significant']:
            return 'significant'
        else:
            return 'minor'

    def detect_progression_trends(self, regional_changes: Dict[str, LongitudinalChange],
                                time_months: float) -> List[ProgressionTrend]:
        """
        Detect progression trends and predict future states

        Args:
            regional_changes: Regional longitudinal changes
            time_months: Time period for analysis

        Returns:
            List of progression trends
        """
        trends = []

        # Analyze hallux valgus progression
        hv_change = regional_changes.get('hallux_valgus_angle') or regional_changes.get('hallux_valgus_hva')
        if hv_change:
            hv_trend = self._analyze_hallux_valgus_progression(hv_change, time_months)
            trends.append(hv_trend)

        # Analyze arch progression
        arch_changes = [change for key, change in regional_changes.items()
                       if 'arch' in key.lower()]
        if arch_changes:
            arch_trend = self._analyze_arch_progression(arch_changes, time_months)
            trends.append(arch_trend)

        return trends

    def _analyze_hallux_valgus_progression(self, change: LongitudinalChange,
                                         time_months: float) -> ProgressionTrend:
        """Analyze hallux valgus progression trend"""
        baseline_severity = self._classify_hva_severity(change.baseline_value)
        followup_severity = self._classify_hva_severity(change.followup_value)

        # Determine progression type
        if change.absolute_change > 2:
            progression_type = 'worsening'
        elif change.absolute_change < -2:
            progression_type = 'improving'
        else:
            progression_type = 'stable'

        # Predict future severity
        rate_per_month = change.rate_per_month
        predicted_6mo = change.followup_value + (rate_per_month * 6)
        predicted_12mo = change.followup_value + (rate_per_month * 12)

        predicted_severity_6mo = self._classify_hva_severity(predicted_6mo)
        predicted_severity_12mo = self._classify_hva_severity(predicted_12mo)

        # Clinical recommendation
        recommendation = self._get_hva_recommendation(
            followup_severity, progression_type, rate_per_month
        )

        return ProgressionTrend(
            condition='Hallux Valgus',
            baseline_severity=baseline_severity,
            followup_severity=followup_severity,
            progression_type=progression_type,
            rate_of_change=rate_per_month,
            predicted_severity_6mo=predicted_severity_6mo,
            predicted_severity_12mo=predicted_severity_12mo,
            clinical_recommendation=recommendation
        )

    def _classify_hva_severity(self, hva_angle: float) -> str:
        """Classify HVA severity"""
        if hva_angle < 15:
            return 'Normal'
        elif hva_angle < 20:
            return 'Mild'
        elif hva_angle < 40:
            return 'Moderate'
        else:
            return 'Severe'

    def _get_hva_recommendation(self, severity: str, progression: str, rate: float) -> str:
        """Get clinical recommendation for HVA"""
        if progression == 'worsening' and rate > 1:
            return 'Urgent orthopaedic consultation recommended'
        elif severity in ['Moderate', 'Severe']:
            return 'Orthopaedic evaluation and intervention planning'
        elif progression == 'worsening':
            return 'Conservative management and monitoring'
        else:
            return 'Continue routine monitoring'

    def _analyze_arch_progression(self, arch_changes: List[LongitudinalChange],
                                time_months: float) -> ProgressionTrend:
        """Analyze arch progression trend"""
        # Use AHI if available, otherwise arch height
        primary_change = None
        for change in arch_changes:
            if 'ahi' in change.region.lower():
                primary_change = change
                break
        if not primary_change and arch_changes:
            primary_change = arch_changes[0]

        if not primary_change:
            return None

        baseline_severity = self._classify_arch_severity(primary_change.baseline_value)
        followup_severity = self._classify_arch_severity(primary_change.followup_value)

        # Determine progression type
        if primary_change.absolute_change > 2:
            progression_type = 'improving' if 'ahi' in primary_change.region.lower() else 'worsening'
        elif primary_change.absolute_change < -2:
            progression_type = 'worsening' if 'ahi' in primary_change.region.lower() else 'improving'
        else:
            progression_type = 'stable'

        return ProgressionTrend(
            condition='Arch Structure',
            baseline_severity=baseline_severity,
            followup_severity=followup_severity,
            progression_type=progression_type,
            rate_of_change=primary_change.rate_per_month,
            predicted_severity_6mo=followup_severity,  # Conservative prediction
            predicted_severity_12mo=followup_severity,
            clinical_recommendation='Monitor arch support and biomechanical function'
        )

    def _classify_arch_severity(self, value: float) -> str:
        """Classify arch severity (assumes AHI if value < 50, otherwise arch height)"""
        if value < 50:  # Likely AHI
            if value < 21:
                return 'Low Arch'
            elif value > 25:
                return 'High Arch'
            else:
                return 'Normal Arch'
        else:  # Likely arch height in mm
            if value < 12:
                return 'Low Arch'
            elif value > 25:
                return 'High Arch'
            else:
                return 'Normal Arch'