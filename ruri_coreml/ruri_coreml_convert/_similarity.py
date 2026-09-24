"""Tiny numeric helper shared by conversion and quantization verification.

Kept separate from both so neither module needs to know the other exists
just to compute a cosine similarity.
"""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)
