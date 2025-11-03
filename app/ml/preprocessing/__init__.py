"""
3D Mesh Preprocessing Module
Medical-grade STL processing and feature extraction
"""

from .stl_processor import STLProcessor, ProcessingResult, QualityMetrics

__all__ = ["STLProcessor", "ProcessingResult", "QualityMetrics"]
