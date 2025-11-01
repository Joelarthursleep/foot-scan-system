"""
STL File Loader for Left and Right Foot Scans
Handles STL format foot scan data
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class STLMeasurements:
    """Basic measurements extracted from STL mesh"""
    foot_length: float
    foot_width: float
    foot_height: float
    volume: float
    surface_area: float
    ball_girth: float = 0.0
    arch_height: float = 0.0
    heel_width: float = 0.0
    instep_girth: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class STLLoader:
    """Load and process STL foot scan files"""

    def __init__(self):
        """Initialize STL loader"""
        self.logger = logging.getLogger(__name__)

    def load_stl_file(self, stl_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load STL file and extract vertices and faces

        Args:
            stl_path: Path to STL file

        Returns:
            Tuple of (vertices, faces)
        """
        try:
            # Load mesh using trimesh
            mesh = trimesh.load(stl_path)

            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError("File does not contain a valid triangular mesh")

            # Extract vertices and faces
            vertices = mesh.vertices.astype(np.float32)
            faces = mesh.faces.astype(np.int32)

            self.logger.info(f"Loaded STL: {len(vertices)} vertices, {len(faces)} faces")

            return vertices, faces

        except Exception as e:
            self.logger.error(f"Failed to load STL file {stl_path}: {e}")
            raise

    def extract_measurements(self, vertices: np.ndarray, faces: np.ndarray) -> STLMeasurements:
        """
        Extract basic measurements from mesh

        Args:
            vertices: Mesh vertices
            faces: Mesh faces

        Returns:
            STLMeasurements object
        """
        try:
            # Create mesh object for calculations
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            # Basic bounding box measurements
            bounds = mesh.bounds
            foot_length = bounds[1][1] - bounds[0][1]  # Y-axis (heel to toe)
            foot_width = bounds[1][0] - bounds[0][0]   # X-axis (medial to lateral)
            foot_height = bounds[1][2] - bounds[0][2]  # Z-axis (bottom to top)

            # Volume and surface area
            volume = mesh.volume
            surface_area = mesh.area

            # Estimated ball girth (circumference at widest point)
            ball_girth = self._estimate_ball_girth(vertices)

            # Estimated arch height
            arch_height = self._estimate_arch_height(vertices)

            # Estimated heel width
            heel_width = self._estimate_heel_width(vertices)

            # Estimated instep girth
            instep_girth = self._estimate_instep_girth(vertices)

            measurements = STLMeasurements(
                foot_length=foot_length,
                foot_width=foot_width,
                foot_height=foot_height,
                volume=volume,
                surface_area=surface_area,
                ball_girth=ball_girth,
                arch_height=arch_height,
                heel_width=heel_width,
                instep_girth=instep_girth,
                metadata={
                    'bounds': bounds.tolist(),
                    'centroid': mesh.centroid.tolist(),
                    'num_vertices': len(vertices),
                    'num_faces': len(faces)
                }
            )

            self.logger.info(f"Extracted measurements - Length: {foot_length:.1f}mm, Width: {foot_width:.1f}mm")

            return measurements

        except Exception as e:
            self.logger.error(f"Failed to extract measurements: {e}")
            raise

    def _estimate_ball_girth(self, vertices: np.ndarray) -> float:
        """Estimate ball girth from vertices"""
        try:
            # Find vertices in the ball area (approximately 65% from heel)
            y_coords = vertices[:, 1]
            y_min, y_max = y_coords.min(), y_coords.max()
            ball_y = y_min + 0.65 * (y_max - y_min)

            # Get vertices near the ball line
            ball_mask = np.abs(y_coords - ball_y) < (y_max - y_min) * 0.05
            ball_vertices = vertices[ball_mask]

            if len(ball_vertices) == 0:
                return 0.0

            # Calculate approximate circumference
            x_coords = ball_vertices[:, 0]
            z_coords = ball_vertices[:, 2]

            width = x_coords.max() - x_coords.min()
            height = z_coords.max() - z_coords.min()

            # Approximate as ellipse circumference
            a, b = width / 2, height / 2
            girth = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))

            return girth

        except Exception:
            return 0.0

    def _estimate_arch_height(self, vertices: np.ndarray) -> float:
        """Estimate arch height"""
        try:
            # Find vertices in the arch area (approximately 50% from heel)
            y_coords = vertices[:, 1]
            y_min, y_max = y_coords.min(), y_coords.max()
            arch_y = y_min + 0.5 * (y_max - y_min)

            # Get vertices near the arch line
            arch_mask = np.abs(y_coords - arch_y) < (y_max - y_min) * 0.05
            arch_vertices = vertices[arch_mask]

            if len(arch_vertices) == 0:
                return 0.0

            # Arch height is difference between highest and lowest points
            z_coords = arch_vertices[:, 2]
            arch_height = z_coords.max() - z_coords.min()

            return arch_height

        except Exception:
            return 0.0

    def _estimate_heel_width(self, vertices: np.ndarray) -> float:
        """Estimate heel width"""
        try:
            # Find vertices in the heel area (first 20% from heel)
            y_coords = vertices[:, 1]
            y_min, y_max = y_coords.min(), y_coords.max()
            heel_y = y_min + 0.2 * (y_max - y_min)

            # Get vertices in heel region
            heel_mask = y_coords <= heel_y
            heel_vertices = vertices[heel_mask]

            if len(heel_vertices) == 0:
                return 0.0

            # Heel width
            x_coords = heel_vertices[:, 0]
            heel_width = x_coords.max() - x_coords.min()

            return heel_width

        except Exception:
            return 0.0

    def _estimate_instep_girth(self, vertices: np.ndarray) -> float:
        """Estimate instep girth"""
        try:
            # Find vertices in the instep area (approximately 40% from heel)
            y_coords = vertices[:, 1]
            y_min, y_max = y_coords.min(), y_coords.max()
            instep_y = y_min + 0.4 * (y_max - y_min)

            # Get vertices near the instep line
            instep_mask = np.abs(y_coords - instep_y) < (y_max - y_min) * 0.05
            instep_vertices = vertices[instep_mask]

            if len(instep_vertices) == 0:
                return 0.0

            # Calculate approximate circumference
            x_coords = instep_vertices[:, 0]
            z_coords = instep_vertices[:, 2]

            width = x_coords.max() - x_coords.min()
            height = z_coords.max() - z_coords.min()

            # Approximate as ellipse circumference
            a, b = width / 2, height / 2
            girth = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))

            return girth

        except Exception:
            return 0.0

    def load_foot_pair(self, left_stl_path: str, right_stl_path: str) -> Dict[str, Any]:
        """
        Load both left and right foot STL files

        Args:
            left_stl_path: Path to left foot STL
            right_stl_path: Path to right foot STL

        Returns:
            Dictionary containing both feet data
        """
        try:
            # Load left foot
            left_vertices, left_faces = self.load_stl_file(left_stl_path)
            left_measurements = self.extract_measurements(left_vertices, left_faces)

            # Load right foot
            right_vertices, right_faces = self.load_stl_file(right_stl_path)
            right_measurements = self.extract_measurements(right_vertices, right_faces)

            # Combine data
            foot_pair_data = {
                'left': {
                    'vertices': left_vertices,
                    'faces': left_faces,
                    'measurements': left_measurements
                },
                'right': {
                    'vertices': right_vertices,
                    'faces': right_faces,
                    'measurements': right_measurements
                },
                'summary': {
                    'avg_length': (left_measurements.foot_length + right_measurements.foot_length) / 2,
                    'avg_width': (left_measurements.foot_width + right_measurements.foot_width) / 2,
                    'avg_volume': (left_measurements.volume + right_measurements.volume) / 2,
                    'length_difference': abs(left_measurements.foot_length - right_measurements.foot_length),
                    'width_difference': abs(left_measurements.foot_width - right_measurements.foot_width)
                }
            }

            self.logger.info(f"Loaded foot pair - Avg length: {foot_pair_data['summary']['avg_length']:.1f}mm")

            return foot_pair_data

        except Exception as e:
            self.logger.error(f"Failed to load foot pair: {e}")
            raise

    def save_temporary_file(self, uploaded_file, suffix: str = "") -> str:
        """
        Save uploaded file to temporary location

        Args:
            uploaded_file: Streamlit uploaded file object
            suffix: Optional suffix for filename

        Returns:
            Path to saved temporary file
        """
        import tempfile

        try:
            # Create temporary file
            temp_dir = Path(tempfile.gettempdir()) / "foot_scan_uploads"
            temp_dir.mkdir(exist_ok=True)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{suffix}_{uploaded_file.name}"
            temp_path = temp_dir / filename

            # Save file
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            self.logger.info(f"Saved temporary file: {temp_path}")

            return str(temp_path)

        except Exception as e:
            self.logger.error(f"Failed to save temporary file: {e}")
            raise

    def analyze_foot_structure(self, vertices: np.ndarray, faces: np.ndarray = None) -> Dict[str, Any]:
        """
        Analyze foot structure for medical conditions with comprehensive measurements

        Args:
            vertices: Mesh vertices
            faces: Mesh faces (optional, for better mesh analysis)

        Returns:
            Dictionary containing structural analysis with medical justifications and regional volume analysis
        """
        try:
            analysis = {}

            # Basic measurements
            bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
            foot_length = bounds[1][1] - bounds[0][1]  # Y-axis
            foot_width = bounds[1][0] - bounds[0][0]   # X-axis
            foot_height = bounds[1][2] - bounds[0][2]  # Z-axis

            # Create mesh for volume calculations (if faces available)
            mesh = None
            if faces is not None:
                try:
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                except:
                    # If mesh creation fails, create a simple mesh using Delaunay triangulation
                    try:
                        from scipy.spatial import Delaunay
                        # Use 2D projection for triangulation (XY plane)
                        points_2d = vertices[:, :2]
                        tri = Delaunay(points_2d)
                        mesh = trimesh.Trimesh(vertices=vertices, faces=tri.simplices)
                    except:
                        mesh = None

            # Enhanced medical condition analysis
            analysis['arch'] = self._analyze_arch_structure(vertices, bounds)
            analysis['instep'] = self._analyze_instep_height(vertices, bounds)
            analysis['alignment'] = self._analyze_foot_alignment(vertices, bounds)

            # New comprehensive bunion analysis
            analysis['bunion'] = self._analyze_bunion_deformity(vertices, bounds)
            analysis['bunionette'] = self._analyze_bunionette_deformity(vertices, bounds)

            # Centerline analysis from ankle to toe box
            analysis['centerline'] = self._analyze_centerline_deviation(vertices, bounds)

            # Big toe angle analysis (Hallux Valgus Angle)
            analysis['hallux_valgus'] = self._analyze_hallux_valgus_angle(vertices, bounds)

            # Intermetatarsal angle analysis
            analysis['intermetatarsal'] = self._analyze_intermetatarsal_angle(vertices, bounds)

            # NEW: Comprehensive regional volume and shape analysis
            if mesh is not None:
                analysis['regional_analysis'] = self._analyze_regional_volumes(mesh, vertices, bounds)
            else:
                # Fallback analysis without mesh
                analysis['regional_analysis'] = self._analyze_regional_volumes_fallback(vertices, bounds)

            # Store basic dimensions
            analysis['dimensions'] = {
                'length': foot_length,
                'width': foot_width,
                'height': foot_height
            }

            # Enhance with clinical measurements and interpretations
            analysis = self._enhance_clinical_measurements(analysis)

            # Advanced condition detection
            analysis['detected_conditions'] = self._detect_foot_conditions(analysis)

            return analysis

        except Exception as e:
            self.logger.error(f"Failed to analyze foot structure: {e}")
            return {}

    def _analyze_arch_structure(self, vertices: np.ndarray, bounds: np.ndarray) -> Dict[str, Any]:
        """Analyze arch height and structure"""
        try:
            y_coords = vertices[:, 1]
            z_coords = vertices[:, 2]

            # Find arch region (middle 40-60% of foot)
            foot_length = bounds[1][1] - bounds[0][1]
            arch_start = bounds[0][1] + 0.4 * foot_length
            arch_end = bounds[0][1] + 0.6 * foot_length

            arch_mask = (y_coords >= arch_start) & (y_coords <= arch_end)
            arch_vertices = vertices[arch_mask]

            if len(arch_vertices) == 0:
                return {'height': 0, 'type': 'unknown'}

            # Calculate arch height
            arch_bottom = arch_vertices[:, 2].min()
            arch_top = arch_vertices[:, 2].max()
            arch_height = arch_top - arch_bottom

            # Determine arch type based on height
            arch_type = 'normal'
            if arch_height > 25.0:  # High arch threshold
                arch_type = 'high'
            elif arch_height < 12.0:  # Flat foot threshold
                arch_type = 'flat'

            # Calculate arch index (ratio of arch height to foot length)
            arch_index = arch_height / (bounds[1][1] - bounds[0][1]) if foot_length > 0 else 0

            # Calculate proper AHI (Arch Height Index) - clinical standard
            # AHI = dorsal arch height / truncated foot length * 100
            # More accurate clinical measurement
            ahi = self._calculate_arch_height_index(vertices, bounds)

            return {
                'height': arch_height,
                'type': arch_type,
                'index': arch_index,
                'ahi': ahi,  # Clinical AHI measurement
                'severity': self._get_arch_severity(arch_height, arch_type)
            }

        except Exception:
            return {'height': 0, 'type': 'unknown'}

    def _analyze_instep_height(self, vertices: np.ndarray, bounds: np.ndarray) -> Dict[str, Any]:
        """Analyze instep height"""
        try:
            y_coords = vertices[:, 1]
            x_coords = vertices[:, 0]
            z_coords = vertices[:, 2]

            # Find instep region (35-45% from heel, central part of foot)
            foot_length = bounds[1][1] - bounds[0][1]
            foot_width = bounds[1][0] - bounds[0][0]

            instep_y = bounds[0][1] + 0.4 * foot_length
            instep_mask = (
                (np.abs(y_coords - instep_y) < foot_length * 0.05) &  # Narrow Y range
                (np.abs(x_coords - (bounds[0][0] + bounds[1][0])/2) < foot_width * 0.3)  # Central X region
            )

            instep_vertices = vertices[instep_mask]

            if len(instep_vertices) == 0:
                return {'height': 0, 'type': 'unknown'}

            # Calculate instep height
            instep_bottom = instep_vertices[:, 2].min()
            instep_top = instep_vertices[:, 2].max()
            instep_height = instep_top - instep_bottom

            # Determine instep type
            instep_type = 'normal'
            if instep_height > 35.0:  # High instep threshold
                instep_type = 'high'
            elif instep_height < 20.0:  # Low instep threshold
                instep_type = 'low'

            return {
                'height': instep_height,
                'type': instep_type,
                'severity': self._get_instep_severity(instep_height, instep_type)
            }

        except Exception:
            return {'height': 0, 'type': 'unknown'}

    def _analyze_foot_alignment(self, vertices: np.ndarray, bounds: np.ndarray) -> Dict[str, Any]:
        """Analyze foot alignment for pronation/supination"""
        try:
            x_coords = vertices[:, 0]
            y_coords = vertices[:, 1]
            z_coords = vertices[:, 2]

            foot_length = bounds[1][1] - bounds[0][1]

            # Analyze heel alignment
            heel_region = y_coords <= (bounds[0][1] + 0.25 * foot_length)
            heel_vertices = vertices[heel_region]

            # Analyze forefoot alignment
            forefoot_region = y_coords >= (bounds[0][1] + 0.65 * foot_length)
            forefoot_vertices = vertices[forefoot_region]

            if len(heel_vertices) == 0 or len(forefoot_vertices) == 0:
                return {'type': 'unknown', 'angle': 0}

            # Calculate medial-lateral balance
            foot_center_x = (bounds[0][0] + bounds[1][0]) / 2

            # Heel alignment
            heel_center_x = heel_vertices[:, 0].mean()
            heel_offset = heel_center_x - foot_center_x

            # Forefoot alignment
            forefoot_center_x = forefoot_vertices[:, 0].mean()
            forefoot_offset = forefoot_center_x - foot_center_x

            # Calculate alignment angle (simplified)
            alignment_angle = np.degrees(np.arctan2(forefoot_offset - heel_offset, foot_length * 0.4))

            # Determine alignment type
            alignment_type = 'neutral'
            if alignment_angle > 3.0:  # Foot angles outward
                alignment_type = 'supination'
            elif alignment_angle < -3.0:  # Foot angles inward
                alignment_type = 'pronation'

            # Calculate severity based on angle magnitude
            severity = 'mild'
            angle_magnitude = abs(alignment_angle)
            if angle_magnitude > 8.0:
                severity = 'severe'
            elif angle_magnitude > 5.0:
                severity = 'moderate'

            return {
                'type': alignment_type,
                'angle': alignment_angle,
                'severity': severity,
                'heel_offset': heel_offset,
                'forefoot_offset': forefoot_offset
            }

        except Exception:
            return {'type': 'unknown', 'angle': 0}

    def _get_arch_severity(self, height: float, arch_type: str) -> str:
        """Determine arch condition severity"""
        if arch_type == 'high':
            if height > 35.0:
                return 'severe'
            elif height > 30.0:
                return 'moderate'
            else:
                return 'mild'
        elif arch_type == 'flat':
            if height < 8.0:
                return 'severe'
            elif height < 10.0:
                return 'moderate'
            else:
                return 'mild'
        return 'normal'

    def _get_instep_severity(self, height: float, instep_type: str) -> str:
        """Determine instep condition severity"""
        if instep_type == 'high':
            if height > 45.0:
                return 'severe'
            elif height > 40.0:
                return 'moderate'
            else:
                return 'mild'
        elif instep_type == 'low':
            if height < 15.0:
                return 'severe'
            elif height < 18.0:
                return 'moderate'
            else:
                return 'mild'
        return 'normal'
    def _analyze_bunion_deformity(self, vertices: np.ndarray, bounds: np.ndarray) -> Dict[str, Any]:
        """
        Analyze bunion deformity (Hallux Valgus) using clinical criteria
        Based on medical research: HVA <15° normal, 15-30° mild, 30-40° moderate, >40° severe
        """
        try:
            foot_length = bounds[1][1] - bounds[0][1]

            # Identify big toe region (first 25% from front of foot)
            y_coords = vertices[:, 1]
            x_coords = vertices[:, 0]
            z_coords = vertices[:, 2]

            toe_region = y_coords >= (bounds[1][1] - 0.25 * foot_length)
            toe_vertices = vertices[toe_region]

            if len(toe_vertices) == 0:
                return {'detected': False, 'angle': 0, 'severity': 'none', 'justification': 'Insufficient toe data'}

            # Find medial prominence (bunion location)
            foot_center_x = (bounds[0][0] + bounds[1][0]) / 2
            medial_side = x_coords < foot_center_x

            # Calculate big toe deviation
            toe_medial_vertices = vertices[toe_region & medial_side]

            if len(toe_medial_vertices) == 0:
                return {'detected': False, 'angle': 0, 'severity': 'none', 'justification': 'No medial toe data'}

            # Estimate hallux valgus angle using geometric analysis
            # Find the most medial point (potential bunion prominence)
            medial_max_x = toe_medial_vertices[:, 0].min()  # Most medial point
            medial_prominence = toe_medial_vertices[toe_medial_vertices[:, 0] == medial_max_x]

            if len(medial_prominence) == 0:
                return {'detected': False, 'angle': 0, 'severity': 'none', 'justification': 'No bunion prominence detected'}

            # Calculate deviation from normal toe alignment
            foot_centerline_x = foot_center_x
            bunion_x = medial_prominence[0][0]
            toe_tip_y = bounds[1][1]  # Front of foot

            # Simplified angle calculation based on medial deviation
            deviation_distance = abs(bunion_x - foot_centerline_x)
            angle_estimate = np.degrees(np.arctan2(deviation_distance, foot_length * 0.2))

            # Clinical classification based on research
            severity = 'none'
            detected = False
            justification = f"Angle: {angle_estimate:.1f}°"

            if angle_estimate >= 15.0:
                detected = True
                if angle_estimate >= 40.0:
                    severity = 'severe'
                    justification += f" - Severe hallux valgus (>40°). Significant medial deviation: {deviation_distance:.1f}mm. Requires immediate attention."
                elif angle_estimate >= 30.0:
                    severity = 'moderate'
                    justification += f" - Moderate hallux valgus (30-40°). Notable bunion prominence. Conservative treatment recommended."
                elif angle_estimate >= 15.0:
                    severity = 'mild'
                    justification += f" - Mild hallux valgus (15-30°). Early bunion formation. Monitor and consider preventive measures."
            else:
                justification += " - Normal toe alignment (<15°). No bunion deformity detected."

            return {
                'detected': detected,
                'angle': angle_estimate,
                'severity': severity,
                'medial_deviation': deviation_distance,
                'prominence_location': [bunion_x, medial_prominence[0][1], medial_prominence[0][2]],
                'justification': justification
            }

        except Exception as e:
            return {'detected': False, 'angle': 0, 'severity': 'none', 'justification': f'Analysis failed: {str(e)}'}

    def _analyze_bunionette_deformity(self, vertices: np.ndarray, bounds: np.ndarray) -> Dict[str, Any]:
        """
        Analyze bunionette (tailor's bunion) - 5th toe deformity
        Similar to bunion but affects the 5th metatarsal
        """
        try:
            foot_length = bounds[1][1] - bounds[0][1]

            # Identify small toe region (lateral side, front 25%)
            y_coords = vertices[:, 1]
            x_coords = vertices[:, 0]

            toe_region = y_coords >= (bounds[1][1] - 0.25 * foot_length)
            foot_center_x = (bounds[0][0] + bounds[1][0]) / 2
            lateral_side = x_coords > foot_center_x

            lateral_toe_vertices = vertices[toe_region & lateral_side]

            if len(lateral_toe_vertices) == 0:
                return {'detected': False, 'angle': 0, 'severity': 'none', 'justification': 'Insufficient lateral toe data'}

            # Find lateral prominence (bunionette location)
            lateral_max_x = lateral_toe_vertices[:, 0].max()  # Most lateral point
            lateral_prominence = lateral_toe_vertices[lateral_toe_vertices[:, 0] == lateral_max_x]

            if len(lateral_prominence) == 0:
                return {'detected': False, 'angle': 0, 'severity': 'none', 'justification': 'No lateral prominence detected'}

            # Calculate lateral deviation
            bunionette_x = lateral_prominence[0][0]
            deviation_distance = abs(bunionette_x - foot_center_x)
            angle_estimate = np.degrees(np.arctan2(deviation_distance, foot_length * 0.2))

            # Classification (similar criteria to bunion)
            severity = 'none'
            detected = False
            justification = f"5th toe angle: {angle_estimate:.1f}°"

            if angle_estimate >= 12.0:  # Slightly lower threshold for bunionette
                detected = True
                if angle_estimate >= 25.0:
                    severity = 'severe'
                    justification += f" - Severe bunionette (>25°). Significant lateral deviation: {deviation_distance:.1f}mm."
                elif angle_estimate >= 18.0:
                    severity = 'moderate'
                    justification += f" - Moderate bunionette (18-25°). Lateral prominence present."
                else:
                    severity = 'mild'
                    justification += f" - Mild bunionette (12-18°). Early 5th toe deformity."
            else:
                justification += " - Normal 5th toe alignment. No bunionette detected."

            return {
                'detected': detected,
                'angle': angle_estimate,
                'severity': severity,
                'lateral_deviation': deviation_distance,
                'prominence_location': [bunionette_x, lateral_prominence[0][1], lateral_prominence[0][2]],
                'justification': justification
            }

        except Exception as e:
            return {'detected': False, 'angle': 0, 'severity': 'none', 'justification': f'Analysis failed: {str(e)}'}

    def _analyze_centerline_deviation(self, vertices: np.ndarray, bounds: np.ndarray) -> Dict[str, Any]:
        """
        Draw centerline from ankle to toe box center and measure deviations
        Essential for pronation/supination analysis
        """
        try:
            foot_length = bounds[1][1] - bounds[0][1]

            # Define ankle region (back 15% of foot)
            ankle_region = vertices[:, 1] <= (bounds[0][1] + 0.15 * foot_length)
            ankle_vertices = vertices[ankle_region]

            # Define toe box region (front 25% of foot)
            toe_region = vertices[:, 1] >= (bounds[1][1] - 0.25 * foot_length)
            toe_vertices = vertices[toe_region]

            if len(ankle_vertices) == 0 or len(toe_vertices) == 0:
                return {'deviation': 0, 'centerline_angle': 0, 'justification': 'Insufficient data for centerline analysis'}

            # Calculate ankle center point
            ankle_center_x = ankle_vertices[:, 0].mean()
            ankle_center_y = ankle_vertices[:, 1].mean()

            # Calculate toe box center point
            toe_center_x = toe_vertices[:, 0].mean()
            toe_center_y = toe_vertices[:, 1].mean()

            # Calculate centerline angle
            centerline_angle = np.degrees(np.arctan2(toe_center_x - ankle_center_x, toe_center_y - ankle_center_y))

            # Calculate foot centerline deviation
            foot_geometric_center_x = (bounds[0][0] + bounds[1][0]) / 2
            centerline_deviation = abs(toe_center_x - ankle_center_x)

            # Analysis
            alignment_type = 'straight'
            severity = 'normal'
            justification = f"Centerline angle: {centerline_angle:.1f}°, deviation: {centerline_deviation:.1f}mm"

            if abs(centerline_angle) > 5.0:
                if centerline_angle > 5.0:
                    alignment_type = 'lateral_deviation'
                    justification += " - Foot deviates laterally, possible supination pattern"
                else:
                    alignment_type = 'medial_deviation'
                    justification += " - Foot deviates medially, possible pronation pattern"

                if abs(centerline_angle) > 12.0:
                    severity = 'severe'
                elif abs(centerline_angle) > 8.0:
                    severity = 'moderate'
                else:
                    severity = 'mild'
            else:
                justification += " - Normal straight foot alignment"

            return {
                'centerline_angle': centerline_angle,
                'deviation': centerline_deviation,
                'ankle_center': [ankle_center_x, ankle_center_y],
                'toe_center': [toe_center_x, toe_center_y],
                'alignment_type': alignment_type,
                'severity': severity,
                'justification': justification
            }

        except Exception as e:
            return {'deviation': 0, 'centerline_angle': 0, 'justification': f'Centerline analysis failed: {str(e)}'}

    def _analyze_hallux_valgus_angle(self, vertices: np.ndarray, bounds: np.ndarray) -> Dict[str, Any]:
        """
        Calculate precise Hallux Valgus Angle (HVA) using clinical methodology
        HVA = angle between 1st metatarsal axis and proximal phalanx axis
        """
        try:
            foot_length = bounds[1][1] - bounds[0][1]

            # Identify 1st metatarsal region (middle section of foot, medial side)
            y_coords = vertices[:, 1]
            x_coords = vertices[:, 0]

            foot_center_x = (bounds[0][0] + bounds[1][0]) / 2
            medial_side = x_coords < foot_center_x

            # 1st metatarsal region (30-70% of foot length from heel)
            metatarsal_region = (y_coords >= bounds[0][1] + 0.3 * foot_length) & (y_coords <= bounds[0][1] + 0.7 * foot_length)
            metatarsal_vertices = vertices[metatarsal_region & medial_side]

            # Proximal phalanx region (70-90% of foot length from heel)
            phalanx_region = (y_coords >= bounds[0][1] + 0.7 * foot_length) & (y_coords <= bounds[0][1] + 0.9 * foot_length)
            phalanx_vertices = vertices[phalanx_region & medial_side]

            if len(metatarsal_vertices) == 0 or len(phalanx_vertices) == 0:
                return {'hva': 0, 'severity': 'none', 'justification': 'Insufficient data for HVA calculation'}

            # Calculate metatarsal axis (linear regression)
            metatarsal_center_x = metatarsal_vertices[:, 0].mean()
            metatarsal_center_y = metatarsal_vertices[:, 1].mean()

            # Calculate phalanx axis
            phalanx_center_x = phalanx_vertices[:, 0].mean()
            phalanx_center_y = phalanx_vertices[:, 1].mean()

            # HVA calculation - angle between metatarsal and phalanx axes
            # Using simplified geometric approach
            metatarsal_slope = 0  # Assuming metatarsal roughly parallel to foot axis
            phalanx_slope = (phalanx_center_x - metatarsal_center_x) / (phalanx_center_y - metatarsal_center_y) if phalanx_center_y != metatarsal_center_y else 0

            # Use enhanced clinical HVA calculation
            hva = self._calculate_hallux_valgus_angle(vertices, bounds)

            # Clinical classification based on medical research
            severity = 'normal'
            justification = f"HVA: {hva:.1f}°"

            if hva >= 15.0:
                if hva >= 40.0:
                    severity = 'severe'
                    justification += " - Severe hallux valgus (≥40°). Surgical intervention may be required."
                elif hva >= 30.0:
                    severity = 'moderate'
                    justification += " - Moderate hallux valgus (30-40°). Conservative treatment recommended."
                elif hva >= 15.0:
                    severity = 'mild'
                    justification += " - Mild hallux valgus (15-30°). Monitor progression."
            else:
                justification += " - Normal big toe alignment (<15°)."

            return {
                'hva': hva,
                'severity': severity,
                'metatarsal_center': [metatarsal_center_x, metatarsal_center_y],
                'phalanx_center': [phalanx_center_x, phalanx_center_y],
                'justification': justification
            }

        except Exception as e:
            return {'hva': 0, 'severity': 'none', 'justification': f'HVA calculation failed: {str(e)}'}

    def _analyze_intermetatarsal_angle(self, vertices: np.ndarray, bounds: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Intermetatarsal Angle (IMA) between 1st and 2nd metatarsals
        IMA >13° indicates significant deformity requiring treatment
        """
        try:
            foot_length = bounds[1][1] - bounds[0][1]
            foot_width = bounds[1][0] - bounds[0][0]
            foot_center_x = (bounds[0][0] + bounds[1][0]) / 2

            y_coords = vertices[:, 1]
            x_coords = vertices[:, 0]

            # Metatarsal region (30-70% from heel)
            metatarsal_region = (y_coords >= bounds[0][1] + 0.3 * foot_length) & (y_coords <= bounds[0][1] + 0.7 * foot_length)

            # 1st metatarsal (medial third)
            medial_boundary = foot_center_x - foot_width * 0.17
            first_mt_region = (x_coords <= medial_boundary) & metatarsal_region
            first_mt_vertices = vertices[first_mt_region]

            # 2nd metatarsal (medial-center region)
            second_mt_left = foot_center_x - foot_width * 0.17
            second_mt_right = foot_center_x - foot_width * 0.05
            second_mt_region = (x_coords >= second_mt_left) & (x_coords <= second_mt_right) & metatarsal_region
            second_mt_vertices = vertices[second_mt_region]

            if len(first_mt_vertices) == 0 or len(second_mt_vertices) == 0:
                return {'ima': 0, 'severity': 'none', 'justification': 'Insufficient metatarsal data for IMA calculation'}

            # Calculate metatarsal axes centers
            first_mt_x = first_mt_vertices[:, 0].mean()
            second_mt_x = second_mt_vertices[:, 0].mean()
            mt_y_center = (first_mt_vertices[:, 1].mean() + second_mt_vertices[:, 1].mean()) / 2

            # IMA calculation - angle between the two metatarsal rays
            ima = abs(np.degrees(np.arctan2(abs(second_mt_x - first_mt_x), foot_length * 0.4)))

            # Clinical classification
            severity = 'normal'
            justification = f"IMA: {ima:.1f}°"

            if ima >= 9.0:
                if ima >= 20.0:
                    severity = 'severe'
                    justification += " - Severe metatarsal deviation (≥20°). Significant forefoot spreading."
                elif ima >= 13.0:
                    severity = 'moderate'
                    justification += " - Moderate IMA (13-20°). Treatment threshold exceeded."
                elif ima >= 9.0:
                    severity = 'mild'
                    justification += " - Mild IMA elevation (9-13°). Monitor progression."
            else:
                justification += " - Normal intermetatarsal angle (<9°)."

            return {
                'ima': ima,
                'severity': severity,
                'first_mt_position': first_mt_x,
                'second_mt_position': second_mt_x,
                'separation': abs(second_mt_x - first_mt_x),
                'justification': justification
            }

        except Exception as e:
            return {'ima': 0, 'severity': 'none', 'justification': f'IMA calculation failed: {str(e)}'}

    def _analyze_regional_volumes(self, mesh: trimesh.Trimesh, vertices: np.ndarray, bounds: np.ndarray) -> Dict[str, Any]:
        """
        Analyze tissue volume and shape characteristics for each anatomical region of the foot

        Returns comprehensive regional analysis including:
        - 9 primary anatomical regions with individual volume analysis
        - Shape characteristics (curvature, surface complexity)
        - Tissue distribution patterns
        - Regional asymmetry detection
        """
        try:
            min_bounds, max_bounds = bounds
            foot_length = max_bounds[0] - min_bounds[0]
            foot_width = max_bounds[1] - min_bounds[1]
            foot_height = max_bounds[2] - min_bounds[2]

            # Define anatomical regions based on clinical foot anatomy
            regions = {
                'hallux': {  # Big toe
                    'x_range': (0.85, 1.0),
                    'y_range': (0.35, 0.65),
                    'clinical_name': 'Hallux (Big Toe)'
                },
                'lesser_toes': {  # Toes 2-5
                    'x_range': (0.8, 1.0),
                    'y_range': (0.0, 0.35),
                    'clinical_name': 'Lesser Toes (2nd-5th)'
                },
                'metatarsal_heads': {  # Ball of foot
                    'x_range': (0.65, 0.85),
                    'y_range': (0.0, 1.0),
                    'clinical_name': 'Metatarsal Heads (Ball of Foot)'
                },
                'midfoot': {  # Arch region
                    'x_range': (0.4, 0.65),
                    'y_range': (0.2, 0.8),
                    'clinical_name': 'Midfoot (Arch Region)'
                },
                'medial_arch': {  # Inside arch
                    'x_range': (0.3, 0.7),
                    'y_range': (0.6, 1.0),
                    'clinical_name': 'Medial Longitudinal Arch'
                },
                'lateral_arch': {  # Outside arch
                    'x_range': (0.3, 0.7),
                    'y_range': (0.0, 0.4),
                    'clinical_name': 'Lateral Longitudinal Arch'
                },
                'heel': {  # Heel region
                    'x_range': (0.0, 0.35),
                    'y_range': (0.25, 0.75),
                    'clinical_name': 'Calcaneus (Heel)'
                },
                'plantar_fascia': {  # Bottom arch connection
                    'x_range': (0.1, 0.8),
                    'y_range': (0.4, 0.6),
                    'clinical_name': 'Plantar Fascia Region'
                },
                'instep': {  # Top of foot
                    'x_range': (0.4, 0.8),
                    'y_range': (0.3, 0.7),
                    'clinical_name': 'Instep (Dorsal Surface)'
                }
            }

            regional_analysis = {}
            total_foot_volume = mesh.volume if hasattr(mesh, 'volume') else 0

            for region_name, region_def in regions.items():
                try:
                    # Extract vertices within regional boundaries
                    x_min = min_bounds[0] + region_def['x_range'][0] * foot_length
                    x_max = min_bounds[0] + region_def['x_range'][1] * foot_length
                    y_min = min_bounds[1] + region_def['y_range'][0] * foot_width
                    y_max = min_bounds[1] + region_def['y_range'][1] * foot_width

                    # Filter vertices in this region
                    region_mask = (
                        (vertices[:, 0] >= x_min) & (vertices[:, 0] <= x_max) &
                        (vertices[:, 1] >= y_min) & (vertices[:, 1] <= y_max)
                    )

                    region_vertices = vertices[region_mask]

                    if len(region_vertices) < 10:  # Insufficient data
                        regional_analysis[region_name] = {
                            'clinical_name': region_def['clinical_name'],
                            'volume_mm3': 0,
                            'volume_percentage': 0,
                            'surface_area_mm2': 0,
                            'vertex_density': 0,
                            'shape_complexity': 0,
                            'height_variation_mm': 0,
                            'curvature_index': 0,
                            'clinical_significance': 'Insufficient data'
                        }
                        continue

                    # Volume calculation (approximation using convex hull)
                    try:
                        region_hull = region_vertices[region_vertices[:, 2].argsort()]  # Sort by height
                        region_volume = self._calculate_regional_volume(region_vertices)
                        volume_percentage = (region_volume / total_foot_volume * 100) if total_foot_volume > 0 else 0
                    except:
                        region_volume = 0
                        volume_percentage = 0

                    # Surface area approximation
                    surface_area = self._calculate_regional_surface_area(region_vertices)

                    # Shape complexity analysis
                    vertex_density = len(region_vertices) / max(1, region_volume / 1000)  # vertices per cm³
                    shape_complexity = self._calculate_shape_complexity(region_vertices)

                    # Height variation (tissue thickness variation)
                    height_variation = np.std(region_vertices[:, 2]) if len(region_vertices) > 1 else 0

                    # Curvature analysis
                    curvature_index = self._calculate_curvature_index(region_vertices)

                    # Clinical assessment
                    clinical_significance = self._assess_regional_clinical_significance(
                        region_name, region_volume, volume_percentage, height_variation, curvature_index
                    )

                    regional_analysis[region_name] = {
                        'clinical_name': region_def['clinical_name'],
                        'volume_mm3': float(region_volume),
                        'volume_percentage': float(volume_percentage),
                        'surface_area_mm2': float(surface_area),
                        'vertex_density': float(vertex_density),
                        'shape_complexity': float(shape_complexity),
                        'height_variation_mm': float(height_variation),
                        'curvature_index': float(curvature_index),
                        'clinical_significance': clinical_significance,
                        'vertex_count': int(len(region_vertices))
                    }

                except Exception as e:
                    regional_analysis[region_name] = {
                        'clinical_name': region_def['clinical_name'],
                        'error': f'Regional analysis failed: {str(e)}',
                        'clinical_significance': 'Analysis failed'
                    }

            # Overall foot shape analysis
            overall_analysis = self._analyze_overall_foot_shape(regional_analysis, total_foot_volume)

            return {
                'regions': regional_analysis,
                'overall': overall_analysis,
                'total_volume_mm3': float(total_foot_volume),
                'analysis_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'error': f'Regional volume analysis failed: {str(e)}',
                'regions': {},
                'overall': {}
            }

    def _calculate_regional_volume(self, vertices: np.ndarray) -> float:
        """Calculate approximate volume for a regional set of vertices"""
        if len(vertices) < 4:
            return 0.0

        try:
            # Use convex hull for volume approximation
            from scipy.spatial import ConvexHull
            hull = ConvexHull(vertices)
            return hull.volume
        except:
            # Fallback: bounding box volume approximation
            min_v = np.min(vertices, axis=0)
            max_v = np.max(vertices, axis=0)
            return np.prod(max_v - min_v)

    def _calculate_regional_surface_area(self, vertices: np.ndarray) -> float:
        """Calculate approximate surface area for a regional set of vertices"""
        if len(vertices) < 3:
            return 0.0

        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(vertices)
            return hull.area
        except:
            # Fallback: approximate using vertex spread
            min_v = np.min(vertices, axis=0)
            max_v = np.max(vertices, axis=0)
            dims = max_v - min_v
            return 2 * (dims[0]*dims[1] + dims[1]*dims[2] + dims[0]*dims[2])

    def _calculate_shape_complexity(self, vertices: np.ndarray) -> float:
        """
        Calculate shape complexity index based on vertex distribution and surface variation
        Higher values indicate more complex, irregular shapes
        """
        if len(vertices) < 5:
            return 0.0

        try:
            # Standard deviation of distances from centroid
            centroid = np.mean(vertices, axis=0)
            distances = np.linalg.norm(vertices - centroid, axis=1)
            distance_variation = np.std(distances) / (np.mean(distances) + 1e-6)

            # Surface roughness approximation
            surface_normals_variation = 0.0
            if len(vertices) > 10:
                # Sample points to estimate surface normal variation
                sample_indices = np.random.choice(len(vertices), min(50, len(vertices)), replace=False)
                sample_vertices = vertices[sample_indices]

                # Calculate local surface normal variations
                normal_variations = []
                for i in range(len(sample_vertices) - 2):
                    v1 = sample_vertices[i+1] - sample_vertices[i]
                    v2 = sample_vertices[i+2] - sample_vertices[i]
                    normal = np.cross(v1, v2)
                    normal_length = np.linalg.norm(normal)
                    if normal_length > 1e-6:
                        normal_variations.append(normal_length)

                if normal_variations:
                    surface_normals_variation = np.std(normal_variations)

            # Combined complexity score (0-10 scale)
            complexity = min(10.0, distance_variation * 5 + surface_normals_variation * 2)
            return complexity

        except Exception:
            return 0.0

    def _calculate_curvature_index(self, vertices: np.ndarray) -> float:
        """
        Calculate curvature index for regional tissue analysis
        Higher values indicate more curved surfaces (important for arch analysis)
        """
        if len(vertices) < 6:
            return 0.0

        try:
            # Principal component analysis to understand surface orientation
            centered_vertices = vertices - np.mean(vertices, axis=0)
            cov_matrix = np.cov(centered_vertices.T)
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending

            # Curvature approximation based on eigenvalue ratios
            if eigenvalues[0] > 1e-6:
                curvature_ratio = eigenvalues[1] / eigenvalues[0]  # Secondary/Primary curvature
                planarity = eigenvalues[2] / eigenvalues[0]       # Tertiary/Primary (flatness)

                # Higher curvature = less planar, more curved
                curvature_index = curvature_ratio * (1 - planarity) * 10
                return min(10.0, curvature_index)

            return 0.0

        except Exception:
            return 0.0

    def _assess_regional_clinical_significance(self, region_name: str, volume: float,
                                             volume_percentage: float, height_variation: float,
                                             curvature_index: float) -> str:
        """Assess clinical significance of regional measurements"""

        # Clinical thresholds based on foot anatomy research
        significance_factors = []

        # Volume-based assessment
        if region_name == 'medial_arch' and volume_percentage < 5:
            significance_factors.append("Low medial arch volume may indicate flat feet")
        elif region_name == 'heel' and volume_percentage > 35:
            significance_factors.append("Large heel volume may indicate heel spurs or swelling")
        elif region_name == 'metatarsal_heads' and volume_percentage > 25:
            significance_factors.append("Enlarged metatarsal region may indicate deformities")

        # Curvature-based assessment
        if region_name in ['medial_arch', 'lateral_arch']:
            if curvature_index < 2:
                significance_factors.append("Reduced arch curvature indicates potential flat feet")
            elif curvature_index > 8:
                significance_factors.append("Excessive arch curvature indicates high arches")

        # Height variation assessment
        if height_variation > 15 and region_name in ['hallux', 'metatarsal_heads']:
            significance_factors.append("High tissue variation may indicate deformities or swelling")

        # Determine overall significance
        if len(significance_factors) >= 2:
            return "High - Multiple indicators present: " + "; ".join(significance_factors)
        elif len(significance_factors) == 1:
            return "Medium - " + significance_factors[0]
        else:
            return "Low - Normal regional characteristics"

    def _analyze_overall_foot_shape(self, regional_analysis: Dict, total_volume: float) -> Dict[str, Any]:
        """Analyze overall foot shape patterns and identify potential conditions"""

        try:
            shape_patterns = []
            diagnostic_indicators = []

            # Extract key regional data
            regions = regional_analysis

            # Arch analysis
            medial_arch = regions.get('medial_arch', {})
            lateral_arch = regions.get('lateral_arch', {})

            if medial_arch.get('curvature_index', 0) < 2 and lateral_arch.get('curvature_index', 0) < 2:
                diagnostic_indicators.append({
                    'condition': 'Flat Feet (Pes Planus)',
                    'confidence': 85,
                    'evidence': 'Low curvature in both medial and lateral arches',
                    'severity': 'moderate'
                })
            elif medial_arch.get('curvature_index', 0) > 8:
                diagnostic_indicators.append({
                    'condition': 'High Arches (Pes Cavus)',
                    'confidence': 80,
                    'evidence': 'Excessive medial arch curvature',
                    'severity': 'mild'
                })

            # Forefoot analysis
            hallux = regions.get('hallux', {})
            metatarsal_heads = regions.get('metatarsal_heads', {})

            if (hallux.get('volume_percentage', 0) > 8 or
                metatarsal_heads.get('height_variation_mm', 0) > 12):
                diagnostic_indicators.append({
                    'condition': 'Hallux Valgus (Bunion)',
                    'confidence': 75,
                    'evidence': 'Enlarged hallux region with increased tissue variation',
                    'severity': 'mild'
                })

            # Heel analysis
            heel = regions.get('heel', {})
            if heel.get('volume_percentage', 0) > 30:
                diagnostic_indicators.append({
                    'condition': 'Heel Enlargement',
                    'confidence': 70,
                    'evidence': 'Disproportionately large heel volume',
                    'severity': 'mild'
                })

            # Overall shape classification
            arch_height_ratio = (medial_arch.get('curvature_index', 0) +
                               lateral_arch.get('curvature_index', 0)) / 2

            if arch_height_ratio < 3:
                foot_type = 'Low Arch (Pronated)'
            elif arch_height_ratio > 7:
                foot_type = 'High Arch (Supinated)'
            else:
                foot_type = 'Normal Arch'

            return {
                'foot_type': foot_type,
                'arch_height_ratio': float(arch_height_ratio),
                'diagnostic_indicators': diagnostic_indicators,
                'total_regions_analyzed': len([r for r in regions.values() if 'volume_mm3' in r]),
                'analysis_quality': 'High' if len(regions) >= 8 else 'Medium'
            }

        except Exception as e:
            return {
                'foot_type': 'Analysis Failed',
                'error': str(e),
                'diagnostic_indicators': [],
                'analysis_quality': 'Failed'
            }

    def _analyze_regional_volumes_fallback(self, vertices: np.ndarray, bounds: np.ndarray) -> Dict[str, Any]:
        """
        Fallback regional analysis when full mesh is not available
        Uses approximate calculations based on vertex distribution
        """
        try:
            min_bounds, max_bounds = bounds
            foot_length = max_bounds[0] - min_bounds[0]
            foot_width = max_bounds[1] - min_bounds[1]
            foot_height = max_bounds[2] - min_bounds[2]

            # Define same anatomical regions
            regions = {
                'hallux': {
                    'x_range': (0.85, 1.0),
                    'y_range': (0.35, 0.65),
                    'clinical_name': 'Hallux (Big Toe)'
                },
                'lesser_toes': {
                    'x_range': (0.8, 1.0),
                    'y_range': (0.0, 0.35),
                    'clinical_name': 'Lesser Toes (2nd-5th)'
                },
                'metatarsal_heads': {
                    'x_range': (0.65, 0.85),
                    'y_range': (0.0, 1.0),
                    'clinical_name': 'Metatarsal Heads (Ball of Foot)'
                },
                'midfoot': {
                    'x_range': (0.4, 0.65),
                    'y_range': (0.2, 0.8),
                    'clinical_name': 'Midfoot (Arch Region)'
                },
                'medial_arch': {
                    'x_range': (0.3, 0.7),
                    'y_range': (0.6, 1.0),
                    'clinical_name': 'Medial Longitudinal Arch'
                },
                'lateral_arch': {
                    'x_range': (0.3, 0.7),
                    'y_range': (0.0, 0.4),
                    'clinical_name': 'Lateral Longitudinal Arch'
                },
                'heel': {
                    'x_range': (0.0, 0.35),
                    'y_range': (0.25, 0.75),
                    'clinical_name': 'Calcaneus (Heel)'
                },
                'plantar_fascia': {
                    'x_range': (0.1, 0.8),
                    'y_range': (0.4, 0.6),
                    'clinical_name': 'Plantar Fascia Region'
                },
                'instep': {
                    'x_range': (0.4, 0.8),
                    'y_range': (0.3, 0.7),
                    'clinical_name': 'Instep (Dorsal Surface)'
                }
            }

            regional_analysis = {}
            total_foot_volume = foot_length * foot_width * foot_height * 0.3  # Approximate foot volume

            for region_name, region_def in regions.items():
                try:
                    # Extract vertices within regional boundaries
                    x_min = min_bounds[0] + region_def['x_range'][0] * foot_length
                    x_max = min_bounds[0] + region_def['x_range'][1] * foot_length
                    y_min = min_bounds[1] + region_def['y_range'][0] * foot_width
                    y_max = min_bounds[1] + region_def['y_range'][1] * foot_width

                    # Filter vertices in this region
                    region_mask = (
                        (vertices[:, 0] >= x_min) & (vertices[:, 0] <= x_max) &
                        (vertices[:, 1] >= y_min) & (vertices[:, 1] <= y_max)
                    )

                    region_vertices = vertices[region_mask]

                    if len(region_vertices) < 5:
                        regional_analysis[region_name] = {
                            'clinical_name': region_def['clinical_name'],
                            'volume_mm3': 0,
                            'volume_percentage': 0,
                            'surface_area_mm2': 0,
                            'vertex_density': 0,
                            'shape_complexity': 0,
                            'height_variation_mm': 0,
                            'curvature_index': 0,
                            'clinical_significance': 'Insufficient data for fallback analysis'
                        }
                        continue

                    # Approximate volume using bounding box
                    region_min = np.min(region_vertices, axis=0)
                    region_max = np.max(region_vertices, axis=0)
                    region_dimensions = region_max - region_min
                    region_volume = np.prod(region_dimensions) * 0.2  # Approximate form factor

                    volume_percentage = (region_volume / total_foot_volume * 100) if total_foot_volume > 0 else 0

                    # Approximate surface area
                    surface_area = 2 * (region_dimensions[0]*region_dimensions[1] +
                                      region_dimensions[1]*region_dimensions[2] +
                                      region_dimensions[0]*region_dimensions[2])

                    # Shape complexity (simplified)
                    vertex_density = len(region_vertices) / max(1, region_volume / 1000)
                    shape_complexity = np.std(np.linalg.norm(region_vertices - np.mean(region_vertices, axis=0), axis=1)) / 10

                    # Height variation
                    height_variation = np.std(region_vertices[:, 2]) if len(region_vertices) > 1 else 0

                    # Simplified curvature
                    curvature_index = min(10.0, height_variation / max(1, np.mean(region_dimensions[:2])) * 50)

                    # Clinical assessment
                    clinical_significance = self._assess_regional_clinical_significance(
                        region_name, region_volume, volume_percentage, height_variation, curvature_index
                    )

                    regional_analysis[region_name] = {
                        'clinical_name': region_def['clinical_name'],
                        'volume_mm3': float(region_volume),
                        'volume_percentage': float(volume_percentage),
                        'surface_area_mm2': float(surface_area),
                        'vertex_density': float(vertex_density),
                        'shape_complexity': float(shape_complexity),
                        'height_variation_mm': float(height_variation),
                        'curvature_index': float(curvature_index),
                        'clinical_significance': clinical_significance,
                        'vertex_count': int(len(region_vertices)),
                        'analysis_method': 'Fallback (Approximate)'
                    }

                except Exception as e:
                    regional_analysis[region_name] = {
                        'clinical_name': region_def['clinical_name'],
                        'error': f'Fallback regional analysis failed: {str(e)}',
                        'clinical_significance': 'Analysis failed'
                    }

            # Overall analysis
            overall_analysis = self._analyze_overall_foot_shape(regional_analysis, total_foot_volume)

            return {
                'regions': regional_analysis,
                'overall': overall_analysis,
                'total_volume_mm3': float(total_foot_volume),
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_method': 'Fallback Analysis (Mesh unavailable)'
            }

        except Exception as e:
            return {
                'error': f'Fallback regional analysis failed: {str(e)}',
                'regions': {},
                'overall': {},
                'analysis_method': 'Failed'
            }

    def _calculate_arch_height_index(self, vertices: np.ndarray, bounds: np.ndarray) -> float:
        """
        Calculate clinical Arch Height Index (AHI)
        AHI = dorsal arch height / truncated foot length * 100

        Standard clinical measurement for foot arch evaluation
        Normal range: 21-25%
        """
        try:
            foot_length = bounds[1][1] - bounds[0][1]  # Y-axis length

            # Find the navicular tuberosity region (50% of foot length)
            navicular_y = bounds[0][1] + 0.5 * foot_length

            # Get cross-section around navicular region (±5mm)
            tolerance = 5.0
            navicular_mask = (vertices[:, 1] >= navicular_y - tolerance) & \
                           (vertices[:, 1] <= navicular_y + tolerance)
            navicular_vertices = vertices[navicular_mask]

            if len(navicular_vertices) == 0:
                return 0.0

            # Find lowest point (plantar surface)
            plantar_height = navicular_vertices[:, 2].min()

            # Find medial-most point at same height level as arch
            medial_vertices = navicular_vertices[navicular_vertices[:, 0] > 0]  # Medial side
            if len(medial_vertices) == 0:
                return 0.0

            # Arch height is the maximum height above plantar surface in arch region
            arch_height = navicular_vertices[:, 2].max() - plantar_height

            # Truncated foot length (heel to ball, approximately 80% of total length)
            truncated_length = 0.8 * foot_length

            # Calculate AHI as percentage
            ahi = (arch_height / truncated_length) * 100 if truncated_length > 0 else 0.0

            return float(ahi)

        except Exception:
            return 0.0

    def _calculate_hallux_valgus_angle(self, vertices: np.ndarray, bounds: np.ndarray) -> float:
        """
        Calculate Hallux Valgus Angle (HVA) - angle of big toe deviation

        Clinical measurement for bunion assessment
        Normal: <15°, Mild: 15-20°, Moderate: 20-40°, Severe: >40°
        """
        try:
            foot_length = bounds[1][1] - bounds[0][1]
            foot_width = bounds[1][0] - bounds[0][0]

            # Find toe region (first 20% of foot length)
            toe_region_end = bounds[0][1] + 0.2 * foot_length
            toe_mask = vertices[:, 1] <= toe_region_end
            toe_vertices = vertices[toe_mask]

            if len(toe_vertices) == 0:
                return 0.0

            # Find big toe (medial-most toe)
            medial_toe_vertices = toe_vertices[toe_vertices[:, 0] > 0]  # Medial side
            if len(medial_toe_vertices) == 0:
                return 0.0

            # Get the most prominent medial point (bunion area)
            max_medial_idx = np.argmax(medial_toe_vertices[:, 0])
            bunion_point = medial_toe_vertices[max_medial_idx]

            # Find metatarsal axis (from midfoot to metatarsal head)
            midfoot_y = bounds[0][1] + 0.5 * foot_length
            midfoot_mask = (vertices[:, 1] >= midfoot_y - 10) & (vertices[:, 1] <= midfoot_y + 10)
            midfoot_vertices = vertices[midfoot_mask]

            if len(midfoot_vertices) == 0:
                return 0.0

            # Center of midfoot
            midfoot_center = np.mean(midfoot_vertices, axis=0)

            # Calculate angle between midfoot center to bunion point and foot axis
            # Foot axis is along Y direction
            bunion_vector = bunion_point - midfoot_center
            foot_axis = np.array([0, 1, 0])  # Y-axis (anterior direction)

            # Calculate angle in degrees
            cos_angle = np.dot(bunion_vector[:2], foot_axis[:2]) / \
                       (np.linalg.norm(bunion_vector[:2]) * np.linalg.norm(foot_axis[:2]))

            # Clamp to valid range for arccos
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)

            # HVA is the deviation from straight (90° - calculated angle)
            hva = abs(90.0 - angle_deg)

            return float(hva)

        except Exception:
            return 0.0

    def _enhance_clinical_measurements(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance existing analysis with additional clinical measurements
        """
        try:
            # Add clinical interpretations
            if 'arch' in analysis and 'ahi' in analysis['arch']:
                ahi = analysis['arch']['ahi']
                if ahi < 21:
                    analysis['arch']['ahi_interpretation'] = 'Low arch (Pes Planus)'
                elif ahi > 25:
                    analysis['arch']['ahi_interpretation'] = 'High arch (Pes Cavus)'
                else:
                    analysis['arch']['ahi_interpretation'] = 'Normal arch'

            if 'hallux_valgus' in analysis and 'angle' in analysis['hallux_valgus']:
                hva = analysis['hallux_valgus']['angle']
                if hva < 15:
                    analysis['hallux_valgus']['severity'] = 'Normal'
                elif hva < 20:
                    analysis['hallux_valgus']['severity'] = 'Mild'
                elif hva < 40:
                    analysis['hallux_valgus']['severity'] = 'Moderate'
                else:
                    analysis['hallux_valgus']['severity'] = 'Severe'

            return analysis

        except Exception:
            return analysis

    def _detect_foot_conditions(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect foot conditions using advanced algorithms

        Args:
            analysis: Complete foot structure analysis

        Returns:
            List of detected conditions with confidence scores
        """
        try:
            # Import the condition detector
            import sys
            import os
            analysis_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'analysis')
            if analysis_dir not in sys.path:
                sys.path.append(analysis_dir)

            from condition_detection import AdvancedConditionDetector

            # Initialize detector
            detector = AdvancedConditionDetector()

            # Detect conditions
            conditions = detector.detect_all_conditions(analysis)

            # Convert to dictionaries for JSON serialization
            condition_dicts = []
            for condition in conditions:
                condition_dict = {
                    'condition_name': condition.condition_name,
                    'severity': condition.severity.value,
                    'confidence_score': condition.confidence_score,
                    'clinical_measurements': condition.clinical_measurements,
                    'detection_criteria': condition.detection_criteria,
                    'clinical_recommendation': condition.clinical_recommendation,
                    'risk_factors': condition.risk_factors,
                    'progression_risk': condition.progression_risk,
                    'treatment_urgency': condition.treatment_urgency
                }
                condition_dicts.append(condition_dict)

            return condition_dicts

        except Exception as e:
            self.logger.error(f"Condition detection failed: {e}")
            return []
