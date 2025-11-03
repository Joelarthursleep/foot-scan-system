"""
3D Mesh Preprocessing Module
Medical-grade STL processing and feature extraction
"""

from .stl_processor import (
    STLProcessor,
    ProcessingResult,
    QualityMetrics,
    ProcessingStatus,
    MorphologicalFeatures as BasicMorphologicalFeatures
)

from .feature_extractor import (
    AdvancedFeatureExtractor,
    CompleteClinicalFeatureSet,
    MorphologicalFeatures,
    BiomechanicalFeatures,
    SymmetryAlignmentFeatures,
    SurfaceAnalysisFeatures,
    ClinicalLandmarkFeatures,
    ClinicalSeverity
)

__all__ = [
    # STL Processing
    "STLProcessor",
    "ProcessingResult",
    "ProcessingStatus",
    "QualityMetrics",

    # Feature Extraction
    "AdvancedFeatureExtractor",
    "CompleteClinicalFeatureSet",
    "MorphologicalFeatures",
    "BiomechanicalFeatures",
    "SymmetryAlignmentFeatures",
    "SurfaceAnalysisFeatures",
    "ClinicalLandmarkFeatures",
    "ClinicalSeverity"
]
