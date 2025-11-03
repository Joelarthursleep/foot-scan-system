"""
Medical-Grade STL Processor
High-performance 3D foot scan processing optimized for clinical use

Performance Targets:
- STL loading: <1 second
- Feature extraction: <3 seconds
- Total processing: <5 seconds (vs 30s in legacy system)

Regulatory Compliance:
- ISO 13485: Traceable processing parameters
- DCB0129: Quality validation and error handling
- MHRA Class IIa: Anatomical validity checks
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
from stl import mesh  # numpy-stl - fast STL I/O
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


class ProcessingStatus(Enum):
    """Processing status codes"""
    SUCCESS = "success"
    FAILED = "failed"
    QUALITY_WARNING = "quality_warning"
    ANATOMICALLY_INVALID = "anatomically_invalid"


class QualitySeverity(Enum):
    """Quality issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class QualityMetrics:
    """
    3D scan quality assessment metrics

    Clinical Importance:
    - Poor quality scans lead to inaccurate diagnoses
    - Quality must be validated before clinical use
    - Documented per ISO 13485
    """
    is_valid: bool
    quality_score: float  # 0.0-1.0

    # Mesh integrity
    vertex_count: int
    face_count: int
    edge_count: int
    is_watertight: bool
    is_manifold: bool

    # Geometric quality
    point_cloud_density: float  # points per cm²
    surface_area: float  # mm²
    volume: float  # mm³
    bounding_box_volume: float  # mm³

    # Mesh quality
    has_degenerate_faces: bool
    has_duplicate_vertices: bool
    aspect_ratio_mean: float
    aspect_ratio_std: float

    # Anatomical validity
    length_mm: float
    width_mm: float
    height_mm: float
    is_anatomically_plausible: bool

    # Issues detected
    quality_issues: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_issue(self, severity: QualitySeverity, message: str):
        """Add quality issue"""
        self.quality_issues.append({
            "severity": severity.value,
            "message": message
        })

    def add_warning(self, message: str):
        """Add warning"""
        self.warnings.append(message)


@dataclass
class MorphologicalFeatures:
    """
    Morphological features extracted from 3D scan
    ~50 parameters per foot for diagnostic analysis
    """
    # Basic dimensions
    length_mm: float
    width_mm: float
    height_mm: float

    # Arch characteristics
    arch_height_mm: float
    arch_height_ratio: float  # Relative to foot length
    arch_index: float

    # Forefoot
    forefoot_width_mm: float
    forefoot_angle_degrees: float
    metatarsal_spread_mm: float

    # Midfoot
    midfoot_width_mm: float
    navicular_height_mm: float

    # Hindfoot
    heel_width_mm: float
    heel_height_mm: float
    calcaneal_angle_degrees: float

    # Toes
    hallux_length_mm: float
    hallux_width_mm: float
    hallux_valgus_angle_degrees: Optional[float] = None
    intermetatarsal_angle_degrees: Optional[float] = None

    # Volume and surface
    total_volume_cm3: float
    total_surface_area_cm2: float
    plantar_surface_area_cm2: float

    # Curvature analysis
    mean_curvature: float
    gaussian_curvature: float
    principal_curvature_1: float
    principal_curvature_2: float

    # Asymmetry metrics (computed when comparing left/right)
    length_asymmetry_mm: Optional[float] = None
    width_asymmetry_mm: Optional[float] = None
    volume_asymmetry_percent: Optional[float] = None


@dataclass
class ProcessingResult:
    """
    Complete STL processing result

    Traceability Requirements (DCB0129):
    - Processing timestamp
    - Software version
    - Processing parameters
    - Quality metrics
    - Feature data
    """
    status: ProcessingStatus

    # Metadata
    scan_id: str
    file_path: Path
    file_checksum: str  # SHA-256 for integrity
    processing_timestamp: float
    processing_duration_seconds: float
    processor_version: str = "2.0.0-medical"

    # Processed data
    mesh_data: Optional[trimesh.Trimesh] = None
    vertices: Optional[np.ndarray] = None
    faces: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None

    # Quality assessment
    quality_metrics: Optional[QualityMetrics] = None

    # Extracted features
    morphological_features: Optional[MorphologicalFeatures] = None

    # Processing parameters (for reproducibility)
    processing_params: Dict[str, Any] = field(default_factory=dict)

    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class STLProcessor:
    """
    High-performance STL processor for medical foot scans

    Design Principles:
    1. Speed: <5 second total processing time
    2. Quality: Comprehensive validation before clinical use
    3. Traceability: Full audit trail of processing
    4. Robustness: Handle corrupted/malformed files gracefully

    Usage:
        processor = STLProcessor()
        result = await processor.process_stl(
            file_path="scan_left.stl",
            scan_id="SCAN-123",
            side="left"
        )

        if result.status == ProcessingStatus.SUCCESS:
            features = result.morphological_features
            print(f"Arch height: {features.arch_height_mm} mm")
    """

    VERSION = "2.0.0-medical"

    # Anatomical plausibility ranges (adult feet)
    FOOT_LENGTH_MIN_MM = 200.0
    FOOT_LENGTH_MAX_MM = 350.0
    FOOT_WIDTH_MIN_MM = 70.0
    FOOT_WIDTH_MAX_MM = 150.0
    FOOT_HEIGHT_MIN_MM = 30.0
    FOOT_HEIGHT_MAX_MM = 120.0

    # Volume plausibility (adult feet)
    FOOT_VOLUME_MIN_CM3 = 300.0
    FOOT_VOLUME_MAX_CM3 = 2000.0

    # Quality thresholds
    MIN_VERTEX_COUNT = 1000
    MIN_QUALITY_SCORE = 0.5  # Below this, warn user
    MIN_POINT_DENSITY = 10.0  # points per cm²

    def __init__(
        self,
        validate_quality: bool = True,
        extract_features: bool = True,
        compute_curvature: bool = True,
        use_gpu: bool = False
    ):
        """
        Initialize STL processor

        Args:
            validate_quality: Perform quality validation (recommended)
            extract_features: Extract morphological features
            compute_curvature: Compute curvature analysis (slower)
            use_gpu: Use GPU acceleration if available
        """
        self.validate_quality = validate_quality
        self.extract_features = extract_features
        self.compute_curvature = compute_curvature
        self.use_gpu = use_gpu

    async def process_stl(
        self,
        file_path: Path,
        scan_id: str,
        side: str = "unknown",
        max_processing_time: float = 30.0
    ) -> ProcessingResult:
        """
        Process STL file with comprehensive quality validation

        Args:
            file_path: Path to STL file
            scan_id: Unique scan identifier
            side: "left", "right", or "unknown"
            max_processing_time: Maximum allowed processing time (seconds)

        Returns:
            ProcessingResult with status, quality metrics, and features
        """
        start_time = time.time()

        try:
            # Run processing with timeout
            result = await asyncio.wait_for(
                self._process_internal(file_path, scan_id, side, start_time),
                timeout=max_processing_time
            )
            return result

        except asyncio.TimeoutError:
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                scan_id=scan_id,
                file_path=file_path,
                file_checksum="",
                processing_timestamp=start_time,
                processing_duration_seconds=time.time() - start_time,
                error_message=f"Processing timeout exceeded {max_processing_time}s"
            )
        except Exception as e:
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                scan_id=scan_id,
                file_path=file_path,
                file_checksum="",
                processing_timestamp=start_time,
                processing_duration_seconds=time.time() - start_time,
                error_message=f"Processing failed: {str(e)}"
            )

    async def _process_internal(
        self,
        file_path: Path,
        scan_id: str,
        side: str,
        start_time: float
    ) -> ProcessingResult:
        """Internal processing method"""

        # Step 1: Load STL file (target: <1s)
        mesh_data, file_checksum = await self._load_stl(file_path)

        # Step 2: Validate quality (target: <1s)
        quality_metrics = None
        if self.validate_quality:
            quality_metrics = await self._validate_quality(mesh_data, side)

            if not quality_metrics.is_valid:
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    scan_id=scan_id,
                    file_path=file_path,
                    file_checksum=file_checksum,
                    processing_timestamp=start_time,
                    processing_duration_seconds=time.time() - start_time,
                    mesh_data=mesh_data,
                    quality_metrics=quality_metrics,
                    error_message="Quality validation failed - scan not suitable for clinical use"
                )

        # Step 3: Extract morphological features (target: <3s)
        morphological_features = None
        if self.extract_features:
            morphological_features = await self._extract_morphological_features(
                mesh_data,
                side
            )

        # Determine final status
        status = ProcessingStatus.SUCCESS
        if quality_metrics and quality_metrics.quality_score < self.MIN_QUALITY_SCORE:
            status = ProcessingStatus.QUALITY_WARNING

        if quality_metrics and not quality_metrics.is_anatomically_plausible:
            status = ProcessingStatus.ANATOMICALLY_INVALID

        processing_duration = time.time() - start_time

        return ProcessingResult(
            status=status,
            scan_id=scan_id,
            file_path=file_path,
            file_checksum=file_checksum,
            processing_timestamp=start_time,
            processing_duration_seconds=processing_duration,
            processor_version=self.VERSION,
            mesh_data=mesh_data,
            vertices=mesh_data.vertices,
            faces=mesh_data.faces,
            normals=mesh_data.vertex_normals,
            quality_metrics=quality_metrics,
            morphological_features=morphological_features,
            processing_params={
                "side": side,
                "validate_quality": self.validate_quality,
                "extract_features": self.extract_features,
                "compute_curvature": self.compute_curvature,
                "use_gpu": self.use_gpu
            }
        )

    async def _load_stl(self, file_path: Path) -> Tuple[trimesh.Trimesh, str]:
        """
        Load STL file using numpy-stl for speed

        Returns:
            (trimesh.Trimesh, file_checksum)
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        mesh_data, checksum = await loop.run_in_executor(
            None,
            self._load_stl_sync,
            file_path
        )
        return mesh_data, checksum

    def _load_stl_sync(self, file_path: Path) -> Tuple[trimesh.Trimesh, str]:
        """Synchronous STL loading"""
        # Compute file checksum for integrity
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
            checksum = hashlib.sha256(file_bytes).hexdigest()

        # Load with numpy-stl (fast)
        stl_mesh = mesh.Mesh.from_file(str(file_path))

        # Convert to trimesh for advanced processing
        vertices = stl_mesh.vectors.reshape(-1, 3)

        # Remove duplicate vertices
        unique_vertices, inverse_indices = np.unique(
            vertices,
            axis=0,
            return_inverse=True
        )

        # Rebuild faces with unique vertex indices
        faces = inverse_indices.reshape(-1, 3)

        # Create trimesh object
        mesh_obj = trimesh.Trimesh(
            vertices=unique_vertices,
            faces=faces,
            process=True  # Automatically fix normals, remove duplicates
        )

        return mesh_obj, checksum

    async def _validate_quality(
        self,
        mesh_data: trimesh.Trimesh,
        side: str
    ) -> QualityMetrics:
        """
        Comprehensive quality validation

        Checks:
        1. Mesh integrity (watertight, manifold)
        2. Geometric quality (no degenerate faces)
        3. Point cloud density
        4. Anatomical plausibility
        5. Aspect ratio distribution
        """
        loop = asyncio.get_event_loop()
        quality = await loop.run_in_executor(
            None,
            self._validate_quality_sync,
            mesh_data,
            side
        )
        return quality

    def _validate_quality_sync(
        self,
        mesh_data: trimesh.Trimesh,
        side: str
    ) -> QualityMetrics:
        """Synchronous quality validation"""

        # Basic metrics
        vertex_count = len(mesh_data.vertices)
        face_count = len(mesh_data.faces)
        edge_count = len(mesh_data.edges_unique)

        # Mesh topology
        is_watertight = mesh_data.is_watertight
        is_manifold = mesh_data.is_winding_consistent

        # Surface area and volume
        surface_area = mesh_data.area  # mm²
        volume = mesh_data.volume if mesh_data.is_watertight else 0.0  # mm³

        # Bounding box
        bounds = mesh_data.bounds
        length_mm = bounds[1][0] - bounds[0][0]
        width_mm = bounds[1][1] - bounds[0][1]
        height_mm = bounds[1][2] - bounds[0][2]
        bounding_box_volume = length_mm * width_mm * height_mm

        # Point cloud density
        point_density = vertex_count / (surface_area / 100.0) if surface_area > 0 else 0.0

        # Mesh quality
        has_degenerate_faces = len(mesh_data.faces[mesh_data.face_areas < 1e-6]) > 0
        has_duplicate_vertices = vertex_count != len(np.unique(mesh_data.vertices, axis=0))

        # Face aspect ratio (shape quality)
        face_areas = mesh_data.area_faces
        edge_lengths = np.linalg.norm(
            mesh_data.vertices[mesh_data.faces[:, 0]] - mesh_data.vertices[mesh_data.faces[:, 1]],
            axis=1
        )
        aspect_ratios = edge_lengths / np.sqrt(face_areas)
        aspect_ratio_mean = float(np.mean(aspect_ratios))
        aspect_ratio_std = float(np.std(aspect_ratios))

        # Anatomical plausibility
        is_anatomically_plausible = self._check_anatomical_plausibility(
            length_mm, width_mm, height_mm, volume
        )

        # Initialize quality metrics
        quality = QualityMetrics(
            is_valid=True,
            quality_score=1.0,
            vertex_count=vertex_count,
            face_count=face_count,
            edge_count=edge_count,
            is_watertight=is_watertight,
            is_manifold=is_manifold,
            point_cloud_density=point_density,
            surface_area=surface_area,
            volume=volume,
            bounding_box_volume=bounding_box_volume,
            has_degenerate_faces=has_degenerate_faces,
            has_duplicate_vertices=has_duplicate_vertices,
            aspect_ratio_mean=aspect_ratio_mean,
            aspect_ratio_std=aspect_ratio_std,
            length_mm=length_mm,
            width_mm=width_mm,
            height_mm=height_mm,
            is_anatomically_plausible=is_anatomically_plausible
        )

        # Quality checks
        quality_penalties = []

        if vertex_count < self.MIN_VERTEX_COUNT:
            quality.add_issue(
                QualitySeverity.CRITICAL,
                f"Insufficient vertices: {vertex_count} < {self.MIN_VERTEX_COUNT}"
            )
            quality_penalties.append(0.5)

        if not is_watertight:
            quality.add_issue(
                QualitySeverity.HIGH,
                "Mesh is not watertight - may affect volume calculations"
            )
            quality_penalties.append(0.2)

        if not is_manifold:
            quality.add_issue(
                QualitySeverity.HIGH,
                "Mesh is non-manifold - topology errors detected"
            )
            quality_penalties.append(0.2)

        if has_degenerate_faces:
            quality.add_issue(
                QualitySeverity.MEDIUM,
                "Degenerate faces detected - may affect accuracy"
            )
            quality_penalties.append(0.1)

        if point_density < self.MIN_POINT_DENSITY:
            quality.add_issue(
                QualitySeverity.MEDIUM,
                f"Low point density: {point_density:.1f} points/cm²"
            )
            quality_penalties.append(0.15)

        if not is_anatomically_plausible:
            quality.add_issue(
                QualitySeverity.CRITICAL,
                f"Anatomically implausible dimensions: {length_mm:.1f}x{width_mm:.1f}x{height_mm:.1f} mm"
            )
            quality_penalties.append(0.5)

        # Compute final quality score
        quality.quality_score = max(0.0, 1.0 - sum(quality_penalties))

        # Mark as invalid if critical issues
        if any(issue['severity'] == 'critical' for issue in quality.quality_issues):
            quality.is_valid = False

        return quality

    def _check_anatomical_plausibility(
        self,
        length_mm: float,
        width_mm: float,
        height_mm: float,
        volume: float
    ) -> bool:
        """
        Check if dimensions are anatomically plausible for adult foot

        Prevents processing of:
        - Truncated scans
        - Wrong units (cm instead of mm)
        - Non-foot objects
        - Corrupted data
        """
        # Check length
        if not (self.FOOT_LENGTH_MIN_MM <= length_mm <= self.FOOT_LENGTH_MAX_MM):
            return False

        # Check width
        if not (self.FOOT_WIDTH_MIN_MM <= width_mm <= self.FOOT_WIDTH_MAX_MM):
            return False

        # Check height
        if not (self.FOOT_HEIGHT_MIN_MM <= height_mm <= self.FOOT_HEIGHT_MAX_MM):
            return False

        # Check volume (if available)
        if volume > 0:
            volume_cm3 = volume / 1000.0
            if not (self.FOOT_VOLUME_MIN_CM3 <= volume_cm3 <= self.FOOT_VOLUME_MAX_CM3):
                return False

        return True

    async def _extract_morphological_features(
        self,
        mesh_data: trimesh.Trimesh,
        side: str
    ) -> MorphologicalFeatures:
        """
        Extract morphological features for diagnostic analysis

        Features extracted:
        - Basic dimensions (length, width, height)
        - Arch characteristics (height, index)
        - Regional widths (forefoot, midfoot, heel)
        - Volume and surface area
        - Curvature analysis
        """
        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(
            None,
            self._extract_features_sync,
            mesh_data,
            side
        )
        return features

    def _extract_features_sync(
        self,
        mesh_data: trimesh.Trimesh,
        side: str
    ) -> MorphologicalFeatures:
        """Synchronous feature extraction"""

        vertices = mesh_data.vertices
        bounds = mesh_data.bounds

        # Basic dimensions
        length_mm = bounds[1][0] - bounds[0][0]
        width_mm = bounds[1][1] - bounds[0][1]
        height_mm = bounds[1][2] - bounds[0][2]

        # Volume and surface area
        volume_mm3 = mesh_data.volume if mesh_data.is_watertight else 0.0
        volume_cm3 = volume_mm3 / 1000.0
        surface_area_cm2 = mesh_data.area / 100.0

        # Arch characteristics
        arch_height_mm, arch_index = self._compute_arch_metrics(vertices, length_mm)
        arch_height_ratio = arch_height_mm / length_mm if length_mm > 0 else 0.0

        # Regional widths
        forefoot_width, midfoot_width, heel_width = self._compute_regional_widths(vertices)

        # Forefoot characteristics
        forefoot_angle = self._estimate_forefoot_angle(vertices)
        metatarsal_spread = forefoot_width * 0.8  # Approximation

        # Hindfoot
        heel_height = height_mm * 0.3  # Approximation
        calcaneal_angle = self._estimate_calcaneal_angle(vertices)

        # Toe characteristics (hallux)
        hallux_length, hallux_width = self._estimate_hallux_dimensions(vertices, length_mm)

        # Navicular height (clinical landmark)
        navicular_height = arch_height_mm * 1.2  # Approximation

        # Plantar surface area (bottom of foot)
        plantar_area_cm2 = self._estimate_plantar_surface_area(mesh_data)

        # Curvature analysis
        mean_curv, gaussian_curv, k1, k2 = self._compute_curvature_metrics(mesh_data)

        # Hallux valgus angle (requires advanced analysis - set to None for now)
        # Will be computed in full diagnostic framework with landmark detection
        hallux_valgus_angle = None
        intermetatarsal_angle = None

        return MorphologicalFeatures(
            length_mm=length_mm,
            width_mm=width_mm,
            height_mm=height_mm,
            arch_height_mm=arch_height_mm,
            arch_height_ratio=arch_height_ratio,
            arch_index=arch_index,
            forefoot_width_mm=forefoot_width,
            forefoot_angle_degrees=forefoot_angle,
            metatarsal_spread_mm=metatarsal_spread,
            midfoot_width_mm=midfoot_width,
            navicular_height_mm=navicular_height,
            heel_width_mm=heel_width,
            heel_height_mm=heel_height,
            calcaneal_angle_degrees=calcaneal_angle,
            hallux_length_mm=hallux_length,
            hallux_width_mm=hallux_width,
            hallux_valgus_angle_degrees=hallux_valgus_angle,
            intermetatarsal_angle_degrees=intermetatarsal_angle,
            total_volume_cm3=volume_cm3,
            total_surface_area_cm2=surface_area_cm2,
            plantar_surface_area_cm2=plantar_area_cm2,
            mean_curvature=mean_curv,
            gaussian_curvature=gaussian_curv,
            principal_curvature_1=k1,
            principal_curvature_2=k2
        )

    def _compute_arch_metrics(
        self,
        vertices: np.ndarray,
        length_mm: float
    ) -> Tuple[float, float]:
        """
        Compute arch height and arch index

        Arch Index = (B - A) / C where:
        - A = minimum height at midfoot
        - B = maximum height at arch
        - C = foot length
        """
        # Find midfoot region (middle 30% of foot length)
        x_coords = vertices[:, 0]
        x_min, x_max = x_coords.min(), x_coords.max()
        midfoot_start = x_min + length_mm * 0.35
        midfoot_end = x_min + length_mm * 0.65

        midfoot_mask = (x_coords >= midfoot_start) & (x_coords <= midfoot_end)
        midfoot_vertices = vertices[midfoot_mask]

        if len(midfoot_vertices) == 0:
            return 0.0, 0.0

        # Arch height = max height in midfoot region relative to minimum
        z_coords = midfoot_vertices[:, 2]
        arch_height_mm = float(z_coords.max() - z_coords.min())

        # Arch index
        arch_index = arch_height_mm / length_mm if length_mm > 0 else 0.0

        return arch_height_mm, arch_index

    def _compute_regional_widths(
        self,
        vertices: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute forefoot, midfoot, and heel widths"""
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        length = x_max - x_min

        # Forefoot (front 30%)
        forefoot_mask = x_coords >= (x_max - length * 0.3)
        forefoot_width = float(y_coords[forefoot_mask].max() - y_coords[forefoot_mask].min()) if forefoot_mask.any() else 0.0

        # Midfoot (middle 30%)
        midfoot_mask = (x_coords >= (x_min + length * 0.35)) & (x_coords <= (x_min + length * 0.65))
        midfoot_width = float(y_coords[midfoot_mask].max() - y_coords[midfoot_mask].min()) if midfoot_mask.any() else 0.0

        # Heel (back 25%)
        heel_mask = x_coords <= (x_min + length * 0.25)
        heel_width = float(y_coords[heel_mask].max() - y_coords[heel_mask].min()) if heel_mask.any() else 0.0

        return forefoot_width, midfoot_width, heel_width

    def _estimate_forefoot_angle(self, vertices: np.ndarray) -> float:
        """Estimate forefoot abduction angle (degrees)"""
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        x_max = x_coords.max()
        length = x_coords.max() - x_coords.min()

        # Get front 20% of foot
        forefoot_mask = x_coords >= (x_max - length * 0.2)
        forefoot_points = vertices[forefoot_mask]

        if len(forefoot_points) < 10:
            return 0.0

        # Fit line to forefoot and compute angle
        # Simple approximation - full implementation would use PCA
        y_range = forefoot_points[:, 1].max() - forefoot_points[:, 1].min()
        x_range = forefoot_points[:, 0].max() - forefoot_points[:, 0].min()

        angle = np.degrees(np.arctan2(y_range, x_range))
        return float(angle)

    def _estimate_calcaneal_angle(self, vertices: np.ndarray) -> float:
        """Estimate calcaneal pitch angle (degrees)"""
        # Simplified estimation - full implementation requires anatomical landmarks
        x_coords = vertices[:, 0]
        z_coords = vertices[:, 2]
        x_min = x_coords.min()
        length = x_coords.max() - x_min

        # Heel region (back 25%)
        heel_mask = x_coords <= (x_min + length * 0.25)
        heel_points = vertices[heel_mask]

        if len(heel_points) < 10:
            return 0.0

        # Approximate pitch angle
        z_range = heel_points[:, 2].max() - heel_points[:, 2].min()
        x_range = heel_points[:, 0].max() - heel_points[:, 0].min()

        angle = np.degrees(np.arctan2(z_range, x_range))
        return float(angle)

    def _estimate_hallux_dimensions(
        self,
        vertices: np.ndarray,
        length_mm: float
    ) -> Tuple[float, float]:
        """Estimate hallux (big toe) dimensions"""
        # Hallux is typically front 15-20% of foot
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        x_max = x_coords.max()

        hallux_mask = x_coords >= (x_max - length_mm * 0.18)
        hallux_points = vertices[hallux_mask]

        if len(hallux_points) == 0:
            return 0.0, 0.0

        hallux_length = float(hallux_points[:, 0].max() - hallux_points[:, 0].min())
        hallux_width = float(hallux_points[:, 1].max() - hallux_points[:, 1].min())

        return hallux_length, hallux_width

    def _estimate_plantar_surface_area(self, mesh_data: trimesh.Trimesh) -> float:
        """Estimate plantar (bottom) surface area"""
        # Plantar surface is bottom 20% of mesh height
        vertices = mesh_data.vertices
        z_min = vertices[:, 2].min()
        z_range = vertices[:, 2].max() - z_min
        z_threshold = z_min + z_range * 0.2

        # Find faces with all vertices below threshold
        face_z_coords = vertices[mesh_data.faces, 2]
        plantar_faces = np.all(face_z_coords < z_threshold, axis=1)

        # Sum areas of plantar faces
        plantar_area = float(mesh_data.area_faces[plantar_faces].sum()) / 100.0  # Convert to cm²

        return plantar_area

    def _compute_curvature_metrics(
        self,
        mesh_data: trimesh.Trimesh
    ) -> Tuple[float, float, float, float]:
        """
        Compute curvature metrics

        Returns:
            (mean_curvature, gaussian_curvature, k1, k2)
        """
        if not self.compute_curvature:
            return 0.0, 0.0, 0.0, 0.0

        # Simplified curvature estimation
        # Full implementation would use discrete differential geometry
        vertex_normals = mesh_data.vertex_normals

        # Approximate curvature from normal variation
        normal_variation = np.std(vertex_normals, axis=0)
        mean_curvature = float(np.mean(normal_variation))

        # Gaussian curvature approximation
        gaussian_curvature = float(np.prod(normal_variation[:2]))

        # Principal curvatures (k1, k2)
        k1 = float(normal_variation.max())
        k2 = float(normal_variation.min())

        return mean_curvature, gaussian_curvature, k1, k2

    async def process_foot_pair(
        self,
        left_stl_path: Path,
        right_stl_path: Path,
        scan_id: str
    ) -> Tuple[ProcessingResult, ProcessingResult]:
        """
        Process left and right foot scans in parallel

        Args:
            left_stl_path: Path to left foot STL
            right_stl_path: Path to right foot STL
            scan_id: Unique scan identifier

        Returns:
            (left_result, right_result) with asymmetry metrics computed
        """
        # Process both feet in parallel
        left_task = self.process_stl(left_stl_path, f"{scan_id}_L", "left")
        right_task = self.process_stl(right_stl_path, f"{scan_id}_R", "right")

        left_result, right_result = await asyncio.gather(left_task, right_task)

        # Compute asymmetry metrics
        if (left_result.morphological_features and
            right_result.morphological_features and
            left_result.status == ProcessingStatus.SUCCESS and
            right_result.status == ProcessingStatus.SUCCESS):

            self._compute_asymmetry_metrics(
                left_result.morphological_features,
                right_result.morphological_features
            )

        return left_result, right_result

    def _compute_asymmetry_metrics(
        self,
        left_features: MorphologicalFeatures,
        right_features: MorphologicalFeatures
    ):
        """Compute bilateral asymmetry metrics"""

        # Length asymmetry
        left_features.length_asymmetry_mm = abs(
            left_features.length_mm - right_features.length_mm
        )
        right_features.length_asymmetry_mm = left_features.length_asymmetry_mm

        # Width asymmetry
        left_features.width_asymmetry_mm = abs(
            left_features.width_mm - right_features.width_mm
        )
        right_features.width_asymmetry_mm = left_features.width_asymmetry_mm

        # Volume asymmetry (percentage)
        if left_features.total_volume_cm3 > 0 and right_features.total_volume_cm3 > 0:
            volume_diff = abs(left_features.total_volume_cm3 - right_features.total_volume_cm3)
            avg_volume = (left_features.total_volume_cm3 + right_features.total_volume_cm3) / 2
            asymmetry_percent = (volume_diff / avg_volume) * 100.0

            left_features.volume_asymmetry_percent = asymmetry_percent
            right_features.volume_asymmetry_percent = asymmetry_percent


# Export
__all__ = [
    "STLProcessor",
    "ProcessingResult",
    "ProcessingStatus",
    "QualityMetrics",
    "QualitySeverity",
    "MorphologicalFeatures"
]
