"""
Volumental Data Loader
Handles loading and parsing of Volumental OBJ files and associated JSON measurements
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

@dataclass
class VolumentalMeasurements:
    """Container for foot measurements from Volumental scan"""
    foot_length: float
    ball_girth: float
    waist_girth: float
    instep_girth: float
    heel_width: float
    toe_heights: Dict[str, float]
    arch_height: float
    scan_id: str
    side: str  # 'left' or 'right'

class VolumentalLoader:
    """Loads and processes Volumental scan data"""

    def __init__(self, obj_path: str, json_path: str):
        """
        Initialize loader with file paths

        Args:
            obj_path: Path to OBJ mesh file
            json_path: Path to JSON measurements file
        """
        self.obj_path = Path(obj_path)
        self.json_path = Path(json_path)

        if not self.obj_path.exists():
            raise FileNotFoundError(f"OBJ file not found: {obj_path}")
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

    def load_measurements(self) -> VolumentalMeasurements:
        """Load measurements from JSON file"""
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        # Extract measurements (adapt keys based on actual Volumental format)
        measurements = VolumentalMeasurements(
            foot_length=data.get('foot_length', 0),
            ball_girth=data.get('ball_girth', 0),
            waist_girth=data.get('waist_girth', 0),
            instep_girth=data.get('instep_girth', 0),
            heel_width=data.get('heel_width', 0),
            toe_heights=data.get('toe_heights', {}),
            arch_height=data.get('arch_height', 0),
            scan_id=data.get('scan_id', 'unknown'),
            side=data.get('side', 'unknown')
        )

        logger.info(f"Loaded measurements for {measurements.side} foot, length: {measurements.foot_length}mm")
        return measurements

    def load_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load 3D mesh from OBJ file

        Returns:
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face indices
        """
        vertices = []
        faces = []

        with open(self.obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    # Parse vertex
                    parts = line.strip().split()
                    vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                    vertices.append(vertex)
                elif line.startswith('f '):
                    # Parse face
                    parts = line.strip().split()
                    # Handle OBJ format: f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
                    face = []
                    for i in range(1, 4):
                        vertex_ref = parts[i].split('/')[0]
                        face.append(int(vertex_ref) - 1)  # OBJ uses 1-based indexing
                    faces.append(face)

        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)

        logger.info(f"Loaded mesh with {len(vertices)} vertices and {len(faces)} faces")
        return vertices, faces

    def get_bounding_box(self, vertices: np.ndarray) -> Dict[str, float]:
        """Calculate bounding box of mesh"""
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)

        return {
            'min': min_coords.tolist(),
            'max': max_coords.tolist(),
            'dimensions': (max_coords - min_coords).tolist(),
            'center': ((min_coords + max_coords) / 2).tolist()
        }

    def validate_data(self, vertices: np.ndarray, measurements: VolumentalMeasurements) -> bool:
        """
        Validate that mesh and measurements are consistent

        Returns:
            True if validation passes
        """
        bbox = self.get_bounding_box(vertices)
        mesh_length = bbox['dimensions'][1]  # Assuming Y is the length axis

        # Check if mesh length roughly matches measured length (within 10%)
        length_diff = abs(mesh_length - measurements.foot_length) / measurements.foot_length

        if length_diff > 0.1:
            logger.warning(f"Mesh length ({mesh_length:.1f}mm) differs from measured length "
                         f"({measurements.foot_length:.1f}mm) by {length_diff*100:.1f}%")
            return False

        return True

    def load_all(self) -> Tuple[np.ndarray, np.ndarray, VolumentalMeasurements]:
        """
        Load both mesh and measurements

        Returns:
            vertices, faces, measurements
        """
        measurements = self.load_measurements()
        vertices, faces = self.load_mesh()

        if not self.validate_data(vertices, measurements):
            logger.warning("Validation failed but continuing...")

        return vertices, faces, measurements


def batch_load_scans(directory: str) -> List[Tuple[str, np.ndarray, np.ndarray, VolumentalMeasurements]]:
    """
    Load all Volumental scans from a directory

    Args:
        directory: Directory containing OBJ and JSON pairs

    Returns:
        List of (scan_id, vertices, faces, measurements)
    """
    scan_dir = Path(directory)
    obj_files = list(scan_dir.glob("*.obj"))

    results = []
    for obj_file in obj_files:
        # Assume JSON has same name as OBJ
        json_file = obj_file.with_suffix('.json')

        if not json_file.exists():
            logger.warning(f"No JSON file found for {obj_file}")
            continue

        try:
            loader = VolumentalLoader(str(obj_file), str(json_file))
            vertices, faces, measurements = loader.load_all()
            scan_id = obj_file.stem
            results.append((scan_id, vertices, faces, measurements))
            logger.info(f"Successfully loaded scan: {scan_id}")
        except Exception as e:
            logger.error(f"Failed to load {obj_file}: {e}")

    return results