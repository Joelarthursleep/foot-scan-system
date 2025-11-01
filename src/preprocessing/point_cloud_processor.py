"""
Point Cloud Processing Module
Converts mesh to point cloud and performs preprocessing
"""

import numpy as np
from typing import Tuple, Optional
import logging
from tqdm import tqdm

from segmentation import get_segmentation_model, PointNetSegmentationModel

logger = logging.getLogger(__name__)

class PointCloudProcessor:
    """Handles mesh to point cloud conversion and preprocessing"""

    def __init__(self, target_points: int = 10000, segmentation_model_path: Optional[str] = None):
        """
        Initialize processor

        Args:
            target_points: Number of points to sample from mesh
        """
        self.target_points = target_points
        self.segmentation_model: Optional[PointNetSegmentationModel] = get_segmentation_model(segmentation_model_path)

    def mesh_to_pointcloud(self, vertices: np.ndarray, faces: np.ndarray,
                          n_points: Optional[int] = None) -> np.ndarray:
        """
        Convert mesh to point cloud by sampling points from surface

        Args:
            vertices: Nx3 array of vertices
            faces: Mx3 array of face indices
            n_points: Number of points to sample (default: self.target_points)

        Returns:
            point_cloud: Nx3 array of sampled points
        """
        if n_points is None:
            n_points = self.target_points

        # Calculate face areas for weighted sampling
        face_areas = self._calculate_face_areas(vertices, faces)

        # Sample faces proportional to their area
        face_probs = face_areas / face_areas.sum()
        sampled_faces = np.random.choice(len(faces), size=n_points, p=face_probs)

        # Sample random points within each selected face
        point_cloud = []
        for face_idx in sampled_faces:
            point = self._sample_point_on_face(vertices, faces[face_idx])
            point_cloud.append(point)

        point_cloud = np.array(point_cloud, dtype=np.float32)
        logger.info(f"Sampled {len(point_cloud)} points from mesh")

        return point_cloud

    def _calculate_face_areas(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Calculate area of each face in mesh"""
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        # Calculate cross product to get area
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)

        return areas

    def _sample_point_on_face(self, vertices: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
        """Sample random point on triangular face using barycentric coordinates"""
        v0 = vertices[face_indices[0]]
        v1 = vertices[face_indices[1]]
        v2 = vertices[face_indices[2]]

        # Generate random barycentric coordinates
        r1, r2 = np.random.random(2)
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2

        # Calculate point using barycentric coordinates
        point = v0 + r1 * (v1 - v0) + r2 * (v2 - v0)
        return point

    def normalize_pointcloud(self, point_cloud: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Normalize point cloud to unit cube centered at origin

        Args:
            point_cloud: Nx3 array of points

        Returns:
            normalized_cloud: Normalized point cloud
            normalization_params: Dictionary with normalization parameters for inverse transform
        """
        # Calculate centroid
        centroid = point_cloud.mean(axis=0)

        # Center the point cloud
        centered = point_cloud - centroid

        # Calculate scale to fit in unit cube
        max_dist = np.max(np.abs(centered))
        scale = 1.0 / max_dist if max_dist > 0 else 1.0

        # Apply scaling
        normalized = centered * scale

        normalization_params = {
            'centroid': centroid,
            'scale': scale
        }

        logger.info(f"Normalized point cloud: centroid={centroid}, scale={scale}")
        return normalized, normalization_params

    def align_foot_orientation(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Align foot to standard coordinate system
        - Y axis: heel to toe (length)
        - X axis: medial to lateral (width)
        - Z axis: bottom to top (height)

        Args:
            point_cloud: Nx3 array of points

        Returns:
            aligned_cloud: Aligned point cloud
        """
        # Use PCA to find principal axes
        centered = point_cloud - point_cloud.mean(axis=0)
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalue (largest first)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Rotate to align with principal axes
        aligned = centered @ eigenvectors.T

        # Ensure heel is at negative Y and bottom is at negative Z
        if aligned[:, 1].min() > aligned[:, 1].max():
            aligned[:, 1] *= -1
        if aligned[:, 2].min() > aligned[:, 2].max():
            aligned[:, 2] *= -1

        return aligned

    def augment_pointcloud(self, point_cloud: np.ndarray,
                          rotation_range: float = 5.0,
                          scale_range: float = 0.02,
                          noise_level: float = 0.001) -> np.ndarray:
        """
        Apply data augmentation to point cloud

        Args:
            point_cloud: Nx3 array of points
            rotation_range: Maximum rotation in degrees
            scale_range: Maximum scale variation
            noise_level: Standard deviation of Gaussian noise

        Returns:
            augmented_cloud: Augmented point cloud
        """
        augmented = point_cloud.copy()

        # Random rotation around Z axis (foot stays flat)
        if rotation_range > 0:
            angle = np.random.uniform(-rotation_range, rotation_range) * np.pi / 180
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            rotation_matrix = np.array([
                [cos_angle, -sin_angle, 0],
                [sin_angle, cos_angle, 0],
                [0, 0, 1]
            ])
            augmented = augmented @ rotation_matrix.T

        # Random scaling
        if scale_range > 0:
            scale = 1.0 + np.random.uniform(-scale_range, scale_range)
            augmented *= scale

        # Add Gaussian noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, augmented.shape)
        augmented += noise

        return augmented

    def segment_pointcloud(self, point_cloud: np.ndarray) -> Optional[np.ndarray]:
        """
        Run anatomical segmentation on the point cloud if a model is available.

        Args:
            point_cloud: Nx3 array of points

        Returns:
            Optional Nx1 array of integer labels.
        """
        if self.segmentation_model is None:
            logger.debug("Segmentation model not available; returning None.")
            return None

        try:
            labels = self.segmentation_model.segment(point_cloud)
            logger.info("Segmentation produced %d unique labels.", len(np.unique(labels)))
            return labels
        except Exception as exc:  # pragma: no cover
            logger.warning("Segmentation failed: %s", exc)
            return None

    def downsample_pointcloud(self, point_cloud: np.ndarray, target_points: int) -> np.ndarray:
        """
        Downsample point cloud to target number of points using farthest point sampling

        Args:
            point_cloud: Nx3 array of points
            target_points: Target number of points

        Returns:
            downsampled: Downsampled point cloud
        """
        if len(point_cloud) <= target_points:
            return point_cloud

        # Farthest point sampling
        selected_indices = [0]  # Start with first point
        distances = np.full(len(point_cloud), np.inf)

        for _ in tqdm(range(target_points - 1), desc="Downsampling"):
            last_point = point_cloud[selected_indices[-1]]

            # Update distances to nearest selected point
            new_distances = np.linalg.norm(point_cloud - last_point, axis=1)
            distances = np.minimum(distances, new_distances)

            # Select point with maximum distance to selected set
            next_idx = np.argmax(distances)
            selected_indices.append(next_idx)

        downsampled = point_cloud[selected_indices]
        return downsampled

    def process_scan(self, vertices: np.ndarray, faces: np.ndarray,
                    normalize: bool = True, align: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Complete processing pipeline for a scan

        Args:
            vertices: Mesh vertices
            faces: Mesh faces
            normalize: Whether to normalize the point cloud
            align: Whether to align foot orientation

        Returns:
            processed_cloud: Processed point cloud
            processing_params: Dictionary with processing parameters
        """
        # Convert mesh to point cloud
        point_cloud = self.mesh_to_pointcloud(vertices, faces)

        processing_params = {}

        # Align orientation
        if align:
            point_cloud = self.align_foot_orientation(point_cloud)
            processing_params['aligned'] = True

        # Normalize
        if normalize:
            point_cloud, norm_params = self.normalize_pointcloud(point_cloud)
            processing_params['normalization'] = norm_params

        processing_params['n_points'] = len(point_cloud)

        return point_cloud, processing_params
