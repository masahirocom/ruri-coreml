"""Loads a saved .mlpackage the way both conversion.py and quantization.py
want it loaded for numerical verification — see
``constants.VERIFICATION_COMPUTE_UNITS_NAME`` for why this forces CPU.
"""

from pathlib import Path

import coremltools as ct

from .constants import VERIFICATION_COMPUTE_UNITS_NAME


def load_for_verification(model_path: Path) -> ct.models.MLModel:
    compute_units = getattr(ct.ComputeUnit, VERIFICATION_COMPUTE_UNITS_NAME)
    return ct.models.MLModel(str(model_path), compute_units=compute_units)
