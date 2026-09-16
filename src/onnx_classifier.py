from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr(frame_bgr: np.ndarray, width: int = 224, height: int = 224) -> np.ndarray:
    """Convert one BGR frame to normalized NCHW float32 input."""
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("frame_bgr must contain image data")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    image = cv2.resize(frame_bgr, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, axis=0).astype(np.float32)


def softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for a one-dimensional score vector."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty 1D vector")
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def top_prediction(probabilities: np.ndarray, labels: Sequence[str]) -> tuple[str, float, int]:
    values = np.asarray(probabilities)
    if values.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if len(labels) != values.size:
        raise ValueError("label count must match model output classes")
    class_id = int(np.argmax(values))
    return labels[class_id], float(values[class_id]), class_id


def read_labels(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        labels = [line.strip() for line in handle if line.strip()]
    if not labels:
        raise ValueError("label file is empty")
    return labels
