"""
PointNet++ segmentation wrapper
Provides optional anatomical labelling for foot point clouds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - optional dependency
    ort = None

LOGGER = logging.getLogger(__name__)


class PointNetSegmentationModel:
    """
    Wrapper around a PointNet++ (or compatible) ONNX segmentation model.

    The model is expected to take an Nx3 point cloud tensor and output
    per-point class probabilities or labels.
    """

    def __init__(self, model_path: Path, device: Optional[str] = None):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed. Install to enable segmentation.")

        if not model_path.exists():
            raise FileNotFoundError(f"Segmentation model not found at {model_path}")

        providers = ["CPUExecutionProvider"]
        if device and "cuda" in device.lower():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        LOGGER.info("Loading segmentation model from %s", model_path)
        self.session = ort.InferenceSession(model_path.as_posix(), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def segment(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Run segmentation on a point cloud.

        Args:
            point_cloud: Nx3 float32 array.

        Returns:
            Array of integer labels per point.
        """
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            raise ValueError("Point cloud must be Nx3.")

        input_tensor = point_cloud.astype(np.float32)[None, ...]  # 1xNx3
        outputs = self.session.run(None, {self.input_name: input_tensor})

        logits = outputs[0]
        if logits.ndim == 3:
            labels = np.argmax(logits, axis=-1)
        else:
            labels = logits

        return labels.squeeze(0).astype(np.int32)


def get_segmentation_model(model_path: Optional[str]) -> Optional[PointNetSegmentationModel]:
    """
    Attempt to create a segmentation model if assets are present.

    Args:
        model_path: Path to ONNX model. If None or missing, returns None.
    """
    if not model_path:
        LOGGER.info("No segmentation model path provided; continuing without segmentation.")
        return None

    path = Path(model_path)
    if not path.exists():
        LOGGER.warning("Segmentation model file not found at %s. Segmentation disabled.", model_path)
        return None

    if ort is None:
        LOGGER.warning("onnxruntime not installed. Install to enable segmentation.")
        return None

    try:
        return PointNetSegmentationModel(path)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("Failed to load segmentation model: %s", exc)
        return None
