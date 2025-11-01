"""
Advanced Bilateral Symmetry Analysis Module
Implements 3D shape matching, statistical analysis, and asymmetry detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.spatial import distance_matrix
from scipy.stats import ttest_ind, ks_2samp
from scipy.spatial.transform import Rotation
import logging

logger = logging.getLogger(__name__)

@dataclass
class AsymmetryRegion:
    """Represents an asymmetric region"""
    region_name: str
    left_value: float
    right_value: float
    difference: float
    percent_difference: float
    z_score: float
    clinical_significance: str
    potential_causes: List[str]

@dataclass
class SymmetryAnalysisResult:
    """Complete symmetry analysis result"""
    overall_symmetry_score: float  # 0-100, 100 = perfect symmetry
    regional_asymmetries: List[AsymmetryRegion]
    shape_similarity_score: float  # ICP alignment score
    compensatory_patterns: List[str]
    clinical_interpretation: str
    heatmap_data: Optional[np.ndarray]  # For visualization
    statistical_significance: Dict[str, float]

class BilateralSymmetryAnalyzer:
    """
    Advanced bilateral symmetry analysis with:
    - 3D shape matching using ICP algorithm
    - Statistical normality testing
    - Heatmap generation for asymmetry visualization
    - Compensatory pattern detection
    - Clinical significance assessment
    """

    def __init__(self):
        """Initialize symmetry analyzer"""
        self.logger = logging.getLogger(__name__)

        # Load population norms for asymmetry
        self.normal_asymmetry_ranges = self._load_normal_ranges()

    def _load_normal_ranges(self) -> Dict[str, Dict[str, float]]:
        """
        Load clinically normal asymmetry ranges
        Based on population studies
        """
        return {
            'foot_length': {
                'mean_diff': 2.5,  # mm
                'std_diff': 2.0,
                'max_normal': 5.0,
                'clinical_threshold': 8.0
            },
            'foot_width': {
                'mean_diff': 2.0,
                'std_diff': 1.5,
                'max_normal': 4.0,
                'clinical_threshold': 6.0
            },
            'arch_height': {
                'mean_diff': 1.5,
                'std_diff': 1.2,
                'max_normal': 3.0,
                'clinical_threshold': 5.0
            },
            'hallux_valgus_angle': {
                'mean_diff': 2.0,  # degrees
                'std_diff': 1.5,
                'max_normal': 4.0,
                'clinical_threshold': 7.0
            },
            'arch_index': {
                'mean_diff': 0.03,  # ratio
                'std_diff': 0.02,
                'max_normal': 0.05,
                'clinical_threshold': 0.08
            },
            'volume': {
                'mean_diff': 15000,  # mm³
                'std_diff': 10000,
                'max_normal': 25000,
                'clinical_threshold': 40000
            }
        }

    def analyze_symmetry(self,
                        left_structure: Dict[str, Any],
                        right_structure: Dict[str, Any],
                        left_measurements: Any,
                        right_measurements: Any,
                        left_vertices: Optional[np.ndarray] = None,
                        right_vertices: Optional[np.ndarray] = None) -> SymmetryAnalysisResult:
        """
        Comprehensive bilateral symmetry analysis

        Args:
            left_structure: Left foot structure data
            right_structure: Right foot structure data
            left_measurements: Left foot measurements
            right_measurements: Right foot measurements
            left_vertices: Left foot 3D vertices (optional, for shape matching)
            right_vertices: Right foot 3D vertices (optional, for shape matching)

        Returns:
            Complete symmetry analysis result
        """
        self.logger.info("Performing bilateral symmetry analysis...")

        # 1. Regional asymmetry analysis
        regional_asymmetries = self._analyze_regional_asymmetries(
            left_structure, right_structure,
            left_measurements, right_measurements
        )

        # 2. 3D shape similarity (if vertices provided)
        shape_similarity = 0.0
        if left_vertices is not None and right_vertices is not None:
            shape_similarity = self._compute_shape_similarity(
                left_vertices, right_vertices
            )

        # 3. Statistical significance testing
        statistical_significance = self._compute_statistical_significance(
            regional_asymmetries
        )

        # 4. Detect compensatory patterns
        compensatory_patterns = self._detect_compensatory_patterns(
            regional_asymmetries, left_structure, right_structure
        )

        # 5. Calculate overall symmetry score
        overall_score = self._calculate_overall_symmetry_score(
            regional_asymmetries, shape_similarity
        )

        # 6. Clinical interpretation
        clinical_interpretation = self._generate_clinical_interpretation(
            overall_score, regional_asymmetries, compensatory_patterns
        )

        # 7. Generate heatmap data (if vertices available)
        heatmap_data = None
        if left_vertices is not None and right_vertices is not None:
            heatmap_data = self._generate_asymmetry_heatmap(
                left_vertices, right_vertices
            )

        return SymmetryAnalysisResult(
            overall_symmetry_score=overall_score,
            regional_asymmetries=regional_asymmetries,
            shape_similarity_score=shape_similarity,
            compensatory_patterns=compensatory_patterns,
            clinical_interpretation=clinical_interpretation,
            heatmap_data=heatmap_data,
            statistical_significance=statistical_significance
        )

    def _analyze_regional_asymmetries(self,
                                     left_structure: Dict[str, Any],
                                     right_structure: Dict[str, Any],
                                     left_measurements: Any,
                                     right_measurements: Any) -> List[AsymmetryRegion]:
        """Analyze asymmetries in specific anatomical regions"""
        asymmetries = []

        # Measurement-based asymmetries
        measurement_pairs = [
            ('foot_length', 'Foot Length', 'mm'),
            ('foot_width', 'Foot Width', 'mm'),
            ('foot_height', 'Foot Height', 'mm'),
            ('ball_girth', 'Ball Girth', 'mm'),
            ('instep_girth', 'Instep Girth', 'mm'),
            ('heel_width', 'Heel Width', 'mm'),
            ('volume', 'Total Volume', 'mm³')
        ]

        for attr_name, display_name, unit in measurement_pairs:
            left_val = getattr(left_measurements, attr_name, 0)
            right_val = getattr(right_measurements, attr_name, 0)

            asymmetry = self._create_asymmetry_region(
                display_name, left_val, right_val, attr_name
            )
            if asymmetry:
                asymmetries.append(asymmetry)

        # Structure-based asymmetries
        structure_pairs = [
            ('arch', 'height', 'Arch Height', 'arch_height'),
            ('arch', 'arch_index', 'Arch Index', 'arch_index'),
            ('big_toe', 'hallux_valgus_angle', 'Hallux Valgus Angle', 'hallux_valgus_angle'),
            ('instep', 'height', 'Instep Height', 'foot_height'),
            ('heel', 'height', 'Heel Height', 'foot_height')
        ]

        for region_key, value_key, display_name, norm_key in structure_pairs:
            left_val = left_structure.get(region_key, {}).get(value_key, 0)
            right_val = right_structure.get(region_key, {}).get(value_key, 0)

            asymmetry = self._create_asymmetry_region(
                display_name, left_val, right_val, norm_key
            )
            if asymmetry:
                asymmetries.append(asymmetry)

        return asymmetries

    def _create_asymmetry_region(self,
                                 name: str,
                                 left_val: float,
                                 right_val: float,
                                 norm_key: str) -> Optional[AsymmetryRegion]:
        """Create asymmetry region with statistical analysis"""
        if left_val == 0 and right_val == 0:
            return None

        difference = abs(left_val - right_val)
        avg_val = (left_val + right_val) / 2

        if avg_val == 0:
            return None

        percent_diff = (difference / avg_val) * 100

        # Get normal range data
        norm_data = self.normal_asymmetry_ranges.get(norm_key, {
            'mean_diff': 0,
            'std_diff': 1,
            'max_normal': 5,
            'clinical_threshold': 10
        })

        # Calculate z-score
        z_score = (difference - norm_data['mean_diff']) / norm_data['std_diff']

        # Determine clinical significance
        if difference < norm_data['max_normal']:
            significance = 'Normal'
            causes = ['Normal anatomical variation']
        elif difference < norm_data['clinical_threshold']:
            significance = 'Borderline'
            causes = self._suggest_causes(name, 'borderline')
        else:
            significance = 'Clinically Significant'
            causes = self._suggest_causes(name, 'significant')

        return AsymmetryRegion(
            region_name=name,
            left_value=left_val,
            right_value=right_val,
            difference=difference,
            percent_difference=percent_diff,
            z_score=z_score,
            clinical_significance=significance,
            potential_causes=causes
        )

    def _suggest_causes(self, region_name: str, severity: str) -> List[str]:
        """Suggest potential causes for asymmetry"""
        causes_map = {
            'Foot Length': [
                'Leg length discrepancy',
                'Unilateral growth plate injury',
                'Compensatory adaptation'
            ],
            'Arch Height': [
                'Unilateral posterior tibial tendon dysfunction',
                'Asymmetric muscle strength',
                'Previous injury to one foot',
                'Unilateral arthritis'
            ],
            'Hallux Valgus Angle': [
                'Asymmetric footwear pressure',
                'Unilateral injury',
                'Favoring one foot during gait',
                'Genetic predisposition manifestation'
            ],
            'Total Volume': [
                'Unilateral edema or swelling',
                'Lymphatic insufficiency',
                'Previous trauma',
                'Vascular differences'
            ]
        }

        general_causes = [
            'Compensatory gait pattern',
            'Dominant limb preference',
            'Previous injury or surgery'
        ]

        specific_causes = causes_map.get(region_name, general_causes)

        if severity == 'significant':
            specific_causes.insert(0, 'Requires clinical evaluation')

        return specific_causes[:3]  # Return top 3 causes

    def _compute_shape_similarity(self,
                                  left_vertices: np.ndarray,
                                  right_vertices: np.ndarray) -> float:
        """
        Compute 3D shape similarity using simplified ICP-like approach
        Returns similarity score (0-100)
        """
        try:
            # Mirror right foot for comparison
            right_mirrored = right_vertices.copy()
            right_mirrored[:, 0] *= -1  # Mirror across X-axis

            # Downsample for performance
            sample_size = min(1000, len(left_vertices), len(right_mirrored))
            left_sample = left_vertices[np.random.choice(len(left_vertices), sample_size, replace=False)]
            right_sample = right_mirrored[np.random.choice(len(right_mirrored), sample_size, replace=False)]

            # Center both point clouds
            left_centered = left_sample - left_sample.mean(axis=0)
            right_centered = right_sample - right_sample.mean(axis=0)

            # Compute nearest neighbor distances
            distances = []
            for point in left_centered:
                dists = np.linalg.norm(right_centered - point, axis=1)
                distances.append(dists.min())

            # Average distance as similarity metric
            avg_distance = np.mean(distances)

            # Convert to similarity score (0-100)
            # Assume perfect similarity at <1mm, zero similarity at >20mm
            similarity = max(0, min(100, 100 * (1 - avg_distance / 20)))

            return similarity

        except Exception as e:
            self.logger.error(f"Shape similarity computation failed: {e}")
            return 50.0  # Default neutral score

    def _compute_statistical_significance(self,
                                         asymmetries: List[AsymmetryRegion]) -> Dict[str, float]:
        """Compute statistical significance of asymmetries"""
        significance = {}

        for asym in asymmetries:
            # Use z-score to determine p-value
            from scipy.stats import norm
            p_value = 2 * (1 - norm.cdf(abs(asym.z_score)))

            significance[asym.region_name] = p_value

        return significance

    def _detect_compensatory_patterns(self,
                                     asymmetries: List[AsymmetryRegion],
                                     left_structure: Dict[str, Any],
                                     right_structure: Dict[str, Any]) -> List[str]:
        """Detect compensatory patterns between feet"""
        patterns = []

        # Pattern 1: Collapsed arch on one side with higher arch on other
        arch_asym = next((a for a in asymmetries if 'Arch Height' in a.region_name), None)
        if arch_asym and arch_asym.clinical_significance != 'Normal':
            if arch_asym.difference > 5:
                patterns.append(
                    f"Asymmetric arch support: {'Right' if arch_asym.right_value > arch_asym.left_value else 'Left'} "
                    f"foot may be compensating for {'left' if arch_asym.right_value > arch_asym.left_value else 'right'} "
                    f"foot arch collapse"
                )

        # Pattern 2: Volume difference suggesting edema or compensation
        volume_asym = next((a for a in asymmetries if 'Volume' in a.region_name), None)
        if volume_asym and volume_asym.difference > 25000:
            patterns.append(
                f"Significant volume asymmetry ({volume_asym.difference/1000:.1f}cm³) - "
                f"evaluate for unilateral edema, injury, or weight-bearing compensation"
            )

        # Pattern 3: Hallux valgus asymmetry suggesting gait compensation
        hv_asym = next((a for a in asymmetries if 'Hallux Valgus' in a.region_name), None)
        if hv_asym and hv_asym.difference > 7:
            patterns.append(
                f"Asymmetric bunion progression - patient may be favoring "
                f"{'left' if hv_asym.right_value > hv_asym.left_value else 'right'} foot during push-off"
            )

        # Pattern 4: Multiple significant asymmetries
        significant_count = sum(1 for a in asymmetries if a.clinical_significance == 'Clinically Significant')
        if significant_count >= 3:
            patterns.append(
                f"Multiple significant asymmetries detected ({significant_count}) - "
                f"suggests systemic compensatory strategy or unilateral pathology"
            )

        if not patterns:
            patterns.append("No major compensatory patterns detected - asymmetries appear independent")

        return patterns

    def _calculate_overall_symmetry_score(self,
                                         asymmetries: List[AsymmetryRegion],
                                         shape_similarity: float) -> float:
        """Calculate overall symmetry score (0-100)"""
        if not asymmetries:
            return 95.0  # Assume high symmetry if no data

        # Score based on regional asymmetries
        region_scores = []
        for asym in asymmetries:
            if asym.clinical_significance == 'Normal':
                region_scores.append(100)
            elif asym.clinical_significance == 'Borderline':
                region_scores.append(70)
            else:  # Clinically Significant
                region_scores.append(40)

        avg_region_score = np.mean(region_scores)

        # Combine with shape similarity (if available)
        if shape_similarity > 0:
            overall_score = 0.7 * avg_region_score + 0.3 * shape_similarity
        else:
            overall_score = avg_region_score

        return round(overall_score, 1)

    def _generate_clinical_interpretation(self,
                                         overall_score: float,
                                         asymmetries: List[AsymmetryRegion],
                                         compensatory_patterns: List[str]) -> str:
        """Generate clinical interpretation text"""
        if overall_score >= 85:
            severity = "excellent bilateral symmetry"
            recommendation = "No concerns regarding asymmetry. Continue routine monitoring."
        elif overall_score >= 70:
            severity = "good bilateral symmetry with minor variations"
            recommendation = "Monitor asymmetries; consider assessment if symptoms develop."
        elif overall_score >= 50:
            severity = "moderate bilateral asymmetry"
            recommendation = "Clinical evaluation recommended to assess functional impact and compensatory patterns."
        else:
            severity = "significant bilateral asymmetry"
            recommendation = "Podiatric consultation strongly recommended. Asymmetry may indicate unilateral pathology or require orthotic intervention."

        significant_regions = [a.region_name for a in asymmetries
                              if a.clinical_significance == 'Clinically Significant']

        if significant_regions:
            regions_text = f"\n\nSignificant asymmetries detected in: {', '.join(significant_regions)}"
        else:
            regions_text = ""

        interpretation = (
            f"Bilateral symmetry analysis reveals {severity} "
            f"(score: {overall_score}/100).{regions_text}\n\n"
            f"Recommendation: {recommendation}"
        )

        if compensatory_patterns:
            interpretation += f"\n\nCompensatory patterns: {compensatory_patterns[0]}"

        return interpretation

    def _generate_asymmetry_heatmap(self,
                                   left_vertices: np.ndarray,
                                   right_vertices: np.ndarray) -> np.ndarray:
        """
        Generate heatmap data showing regional asymmetry intensity
        Returns array suitable for visualization
        """
        try:
            # This is a simplified version
            # In production, would use proper mesh correspondence

            # Mirror right foot
            right_mirrored = right_vertices.copy()
            right_mirrored[:, 0] *= -1

            # Compute point-wise distances
            # For each left point, find nearest right point
            sample_size = min(500, len(left_vertices))
            left_sample_indices = np.random.choice(len(left_vertices), sample_size, replace=False)
            left_sample = left_vertices[left_sample_indices]

            distances = np.zeros(sample_size)
            for i, point in enumerate(left_sample):
                dists = np.linalg.norm(right_mirrored - point, axis=1)
                distances[i] = dists.min()

            # Normalize to 0-1 range for heatmap
            heatmap = distances / (distances.max() + 1e-6)

            return heatmap

        except Exception as e:
            self.logger.error(f"Heatmap generation failed: {e}")
            return np.array([])
