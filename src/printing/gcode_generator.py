"""
3D Printing Generation Module
Generates G-code for TPU composite material application on shoe lasts
"""

import numpy as np
import trimesh
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PrintSettings:
    """3D printing parameters for TPU"""
    layer_height: float = 0.2  # mm
    nozzle_temp: int = 230  # °C
    bed_temp: int = 60  # °C
    print_speed: int = 30  # mm/s
    infill_density: float = 0.4  # 40%
    retraction_distance: float = 2.0  # mm
    retraction_speed: int = 40  # mm/s
    shore_hardness: str = "85A"

@dataclass
class PrintJob:
    """Complete print job information"""
    gcode_path: str
    stl_path: str
    estimated_time_minutes: float
    material_volume_cm3: float
    material_weight_g: float
    layer_count: int
    success_probability: float

class BuildupGenerator:
    """Generates 3D buildup geometry for last customization"""

    def __init__(self):
        self.settings = PrintSettings()

    def generate_buildup_mesh(self,
                             point_cloud: np.ndarray,
                             thickness_map: np.ndarray,
                             base_last_mesh: Optional[trimesh.Trimesh] = None) -> trimesh.Trimesh:
        """
        Generate 3D mesh of material buildup

        Args:
            point_cloud: Foot scan point cloud
            thickness_map: Required thickness at each point
            base_last_mesh: Optional base last geometry

        Returns:
            Trimesh object of buildup geometry
        """
        # Filter points that need buildup
        buildup_indices = np.where(np.abs(thickness_map) > 0.1)[0]

        if len(buildup_indices) == 0:
            logger.warning("No buildup required")
            return None

        buildup_points = point_cloud[buildup_indices]
        buildup_thickness = thickness_map[buildup_indices]

        # Create offset points for buildup volume
        offset_points = []
        for point, thickness in zip(buildup_points, buildup_thickness):
            # Calculate normal direction (simplified: use local neighborhood)
            normal = self._estimate_normal(point, buildup_points)

            # Create offset point
            offset_point = point + normal * thickness
            offset_points.append(offset_point)

        offset_points = np.array(offset_points)

        # Combine original and offset points
        all_points = np.vstack([buildup_points, offset_points])

        # Create mesh using Delaunay triangulation
        try:
            buildup_mesh = self._create_mesh_from_points(all_points)
            buildup_mesh = self._smooth_mesh(buildup_mesh)

            logger.info(f"Generated buildup mesh with {len(buildup_mesh.vertices)} vertices")
            return buildup_mesh

        except Exception as e:
            logger.error(f"Failed to generate buildup mesh: {e}")
            return None

    def _estimate_normal(self, point: np.ndarray,
                        neighborhood: np.ndarray,
                        k: int = 10) -> np.ndarray:
        """Estimate surface normal at a point"""
        # Find k nearest neighbors
        distances = np.linalg.norm(neighborhood - point, axis=1)
        nearest_indices = np.argsort(distances)[1:k+1]  # Exclude self

        if len(nearest_indices) < 3:
            return np.array([0, 0, 1])  # Default upward normal

        # Fit plane using PCA
        neighbors = neighborhood[nearest_indices]
        centered = neighbors - neighbors.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Normal is eigenvector with smallest eigenvalue
        normal = eigvecs[:, np.argmin(eigvals)]

        # Ensure normal points outward (positive Z component)
        if normal[2] < 0:
            normal = -normal

        return normal / np.linalg.norm(normal)

    def _create_mesh_from_points(self, points: np.ndarray) -> trimesh.Trimesh:
        """Create mesh from point cloud using convex hull"""
        from scipy.spatial import ConvexHull

        hull = ConvexHull(points)
        mesh = trimesh.Trimesh(
            vertices=points,
            faces=hull.simplices
        )

        return mesh

    def _smooth_mesh(self, mesh: trimesh.Trimesh,
                    iterations: int = 2) -> trimesh.Trimesh:
        """Apply smoothing to mesh"""
        for _ in range(iterations):
            mesh = mesh.smoothed()

        return mesh

class GcodeGenerator:
    """Generates G-code for 3D printing TPU on lasts"""

    def __init__(self, settings: Optional[PrintSettings] = None):
        """Initialize with print settings"""
        self.settings = settings or PrintSettings()

    def generate_gcode(self, buildup_mesh: trimesh.Trimesh,
                      output_path: str,
                      zones: Optional[Dict[str, Dict]] = None) -> PrintJob:
        """
        Generate G-code from buildup mesh

        Args:
            buildup_mesh: Mesh to print
            output_path: Path to save G-code
            zones: Optional zone-specific settings (e.g., different infill)

        Returns:
            PrintJob object with job details
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export mesh as STL first
        stl_path = output_path.with_suffix('.stl')
        buildup_mesh.export(str(stl_path))

        # Generate G-code
        gcode_lines = []

        # Add header
        gcode_lines.extend(self._generate_header())

        # Generate layers
        layers = self._slice_mesh(buildup_mesh)

        for layer_num, layer in enumerate(layers):
            gcode_lines.append(f"; Layer {layer_num}")
            gcode_lines.extend(self._generate_layer(layer, layer_num))

        # Add footer
        gcode_lines.extend(self._generate_footer())

        # Write G-code file
        with open(output_path, 'w') as f:
            f.write('\n'.join(gcode_lines))

        # Calculate job statistics
        job = self._calculate_job_stats(buildup_mesh, layers, str(output_path))

        logger.info(f"Generated G-code: {output_path}")
        logger.info(f"Estimated print time: {job.estimated_time_minutes:.1f} minutes")
        logger.info(f"Material required: {job.material_volume_cm3:.1f} cm³")

        return job

    def _generate_header(self) -> List[str]:
        """Generate G-code header"""
        header = [
            "; Generated by Foot Scan to Custom Last System",
            "; Material: TPU",
            f"; Layer height: {self.settings.layer_height}mm",
            f"; Nozzle temp: {self.settings.nozzle_temp}°C",
            f"; Bed temp: {self.settings.bed_temp}°C",
            "",
            "G21 ; Metric units",
            "G90 ; Absolute positioning",
            "M82 ; Absolute extrusion",
            f"M104 S{self.settings.nozzle_temp} ; Set nozzle temp",
            f"M140 S{self.settings.bed_temp} ; Set bed temp",
            "M109 ; Wait for nozzle temp",
            "M190 ; Wait for bed temp",
            "G28 ; Home all axes",
            "G92 E0 ; Reset extruder",
            ""
        ]
        return header

    def _generate_footer(self) -> List[str]:
        """Generate G-code footer"""
        footer = [
            "",
            "M104 S0 ; Turn off nozzle",
            "M140 S0 ; Turn off bed",
            "M84 ; Disable motors",
            "M107 ; Turn off fan",
            "; End of print"
        ]
        return footer

    def _slice_mesh(self, mesh: trimesh.Trimesh) -> List[np.ndarray]:
        """Slice mesh into layers"""
        # Get Z bounds
        z_min = mesh.vertices[:, 2].min()
        z_max = mesh.vertices[:, 2].max()

        # Generate layer heights
        layer_heights = np.arange(z_min, z_max, self.settings.layer_height)

        layers = []
        for z in layer_heights:
            # Get cross-section at this Z height
            slice_2d = mesh.section(plane_origin=[0, 0, z],
                                   plane_normal=[0, 0, 1])

            if slice_2d is not None:
                try:
                    # Convert to 2D path
                    path = slice_2d.to_planar()[0]
                    layers.append(path)
                except:
                    pass

        return layers

    def _generate_layer(self, layer_path, layer_num: int) -> List[str]:
        """Generate G-code for a single layer"""
        gcode = []

        z_height = layer_num * self.settings.layer_height

        # Move to layer height
        gcode.append(f"G0 Z{z_height:.3f}")

        # Generate perimeter
        if hasattr(layer_path, 'vertices'):
            vertices = layer_path.vertices

            # Move to start position
            gcode.append(f"G0 X{vertices[0][0]:.3f} Y{vertices[0][1]:.3f}")

            # Print perimeter
            for vertex in vertices[1:]:
                gcode.append(f"G1 X{vertex[0]:.3f} Y{vertex[1]:.3f} "
                           f"F{self.settings.print_speed * 60}")

            # Generate infill
            infill = self._generate_infill(vertices, self.settings.infill_density)
            gcode.extend(infill)

        return gcode

    def _generate_infill(self, perimeter: np.ndarray,
                        density: float) -> List[str]:
        """Generate infill pattern"""
        gcode = []

        if len(perimeter) < 3:
            return gcode

        # Simple line infill pattern
        x_min = perimeter[:, 0].min()
        x_max = perimeter[:, 0].max()
        y_min = perimeter[:, 1].min()
        y_max = perimeter[:, 1].max()

        # Calculate line spacing based on density
        spacing = 1.0 / density  # mm

        # Generate horizontal lines
        y = y_min
        direction = 1
        while y < y_max:
            if direction == 1:
                gcode.append(f"G1 X{x_max:.3f} Y{y:.3f}")
            else:
                gcode.append(f"G1 X{x_min:.3f} Y{y:.3f}")
            direction *= -1
            y += spacing

        return gcode

    def _calculate_job_stats(self, mesh: trimesh.Trimesh,
                           layers: List, gcode_path: str) -> PrintJob:
        """Calculate print job statistics"""
        # Volume calculation
        volume_mm3 = mesh.volume if mesh.is_watertight else mesh.convex_hull.volume
        volume_cm3 = volume_mm3 / 1000

        # Weight (TPU density ~1.2 g/cm³)
        weight_g = volume_cm3 * 1.2

        # Time estimation (simplified)
        total_distance = 0
        for layer in layers:
            if hasattr(layer, 'vertices') and len(layer.vertices) > 1:
                distances = np.linalg.norm(
                    layer.vertices[1:] - layer.vertices[:-1], axis=1
                )
                total_distance += distances.sum()

        print_time_seconds = total_distance / self.settings.print_speed
        print_time_minutes = print_time_seconds / 60

        # Add layer change time
        print_time_minutes += len(layers) * 0.1  # 6 seconds per layer

        return PrintJob(
            gcode_path=gcode_path,
            stl_path=str(Path(gcode_path).with_suffix('.stl')),
            estimated_time_minutes=print_time_minutes,
            material_volume_cm3=volume_cm3,
            material_weight_g=weight_g,
            layer_count=len(layers),
            success_probability=0.95  # Simplified
        )

def generate_custom_last_modifications(
    foot_scan: np.ndarray,
    segmentation: np.ndarray,
    features: Dict,
    output_dir: str
) -> PrintJob:
    """
    Complete pipeline to generate 3D printing files

    Args:
        foot_scan: Point cloud of foot
        segmentation: Segmentation labels
        features: Detected anatomical features
        output_dir: Output directory

    Returns:
        PrintJob with all printing information
    """
    # Calculate thickness map based on features
    thickness_map = np.zeros(len(foot_scan))

    # Add bunion relief
    if features.get('bunion', {}).get('has_bunion'):
        bunion_indices = features['bunion'].get('affected_points', [])
        severity = features['bunion'].get('severity', 'mild')

        relief_depth = {'mild': -2, 'moderate': -3.5, 'severe': -5}[severity]
        thickness_map[bunion_indices] = relief_depth

    # Add arch support
    if features.get('arch', {}).get('support_level_needed') != 'minimal':
        arch_mask = segmentation == 10  # Medial arch segment
        support_height = {'moderate': 3, 'maximum': 5}[
            features['arch']['support_level_needed']
        ]
        thickness_map[arch_mask] = support_height

    # Generate buildup mesh
    generator = BuildupGenerator()
    buildup_mesh = generator.generate_buildup_mesh(foot_scan, thickness_map)

    if buildup_mesh is None:
        logger.warning("No modifications needed")
        return None

    # Generate G-code
    gcode_gen = GcodeGenerator()
    output_path = Path(output_dir) / "custom_last_modification.gcode"

    job = gcode_gen.generate_gcode(buildup_mesh, str(output_path))

    return job