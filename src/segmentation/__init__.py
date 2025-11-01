"""
Segmentation package for point-cloud based anatomical labelling.
"""

from .pointnet_segmenter import PointNetSegmentationModel, get_segmentation_model

__all__ = ["PointNetSegmentationModel", "get_segmentation_model"]
