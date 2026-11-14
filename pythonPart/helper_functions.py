from __future__ import annotations

import numpy as np


def normalize(landmarks_21x3: np.ndarray) -> np.ndarray:
    pts = np.asarray(landmarks_21x3, dtype=np.float32)
    if pts.shape != (21, 3):
        raise ValueError(f"Expected (21,3), got {pts.shape}")
    wrist = pts[0].copy()
    centered = pts - wrist
    scale = np.linalg.norm(pts[9, :2] - pts[0, :2])
    if scale < 1e-6:
        scale = 1e-6
    return (centered / scale).reshape(-1)


def to_media_pipe(label: str) -> str:
    if not label:
        return "None"
    if label == "none":
        return "None"
    parts = [p for p in str(label).strip().split("_") if p]
    if not parts:
        return "None"
    return "_".join(p.capitalize() for p in parts)
