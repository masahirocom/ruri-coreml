"""int8 weight quantization of an already-converted ruri-v3 Core ML model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ._similarity import cosine_similarity
from ._verification_loading import load_for_verification
from .constants import (
    COREML_ATTENTION_MASK_FEATURE_NAME,
    COREML_INPUT_IDS_FEATURE_NAME,
    COREML_OUTPUT_FEATURE_NAME,
    INT8_QUANTIZATION_DTYPE,
    INT8_QUANTIZATION_MODE,
    INT8_QUANTIZATION_WEIGHT_THRESHOLD,
    VERIFICATION_PREFIX,
    VERIFICATION_TEXT,
)


@dataclass(frozen=True)
class QuantizationRequest:
    source_model_path: Path
    destination_model_path: Path
    tokenizer_huggingface_id: str
    sequence_length: int


@dataclass(frozen=True)
class QuantizationResult:
    destination_path: Path
    quantized_vs_source_cosine_similarity: float


def quantize_int8(request: QuantizationRequest) -> QuantizationResult:
    source_model = ct.models.MLModel(str(request.source_model_path))
    quantized_model = cto.linear_quantize_weights(source_model, config=_quantization_config())
    quantized_model.save(str(request.destination_model_path))

    tokenizer = AutoTokenizer.from_pretrained(request.tokenizer_huggingface_id)
    example_inputs = _build_example_inputs(tokenizer, request.sequence_length)

    # Reload both from disk under forced-CPU verification settings rather
    # than reusing the in-memory objects above — see load_for_verification's
    # docstring for why that matters on some hosts.
    source_output = load_for_verification(request.source_model_path).predict(
        example_inputs
    )[COREML_OUTPUT_FEATURE_NAME][0]
    quantized_output = load_for_verification(request.destination_model_path).predict(
        example_inputs
    )[COREML_OUTPUT_FEATURE_NAME][0]

    return QuantizationResult(
        destination_path=request.destination_model_path,
        quantized_vs_source_cosine_similarity=cosine_similarity(quantized_output, source_output),
    )


def _quantization_config() -> cto.OptimizationConfig:
    op_config = cto.OpLinearQuantizerConfig(
        mode=INT8_QUANTIZATION_MODE,
        dtype=INT8_QUANTIZATION_DTYPE,
        weight_threshold=INT8_QUANTIZATION_WEIGHT_THRESHOLD,
    )
    return cto.OptimizationConfig(global_config=op_config)


def _build_example_inputs(tokenizer: PreTrainedTokenizerBase, sequence_length: int) -> dict[str, np.ndarray]:
    encoded = tokenizer(
        [VERIFICATION_PREFIX.value + VERIFICATION_TEXT],
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
    )
    return {
        COREML_INPUT_IDS_FEATURE_NAME: encoded["input_ids"].astype(np.int32),
        COREML_ATTENTION_MASK_FEATURE_NAME: encoded["attention_mask"].astype(np.int32),
    }
