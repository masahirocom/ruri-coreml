"""PyTorch -> Core ML conversion for a ruri-v3 checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from . import modernbert_mask_patch
from ._similarity import cosine_similarity
from ._verification_loading import load_for_verification
from .constants import (
    COREML_ATTENTION_MASK_FEATURE_NAME,
    COREML_INPUT_IDS_FEATURE_NAME,
    COREML_OUTPUT_FEATURE_NAME,
    EXPORT_BATCH_SIZE,
    MINIMUM_DEPLOYMENT_TARGET_NAME,
    MODEL_ATTENTION_IMPLEMENTATION,
    VERIFICATION_PREFIX,
    VERIFICATION_TEXT,
    ComputePrecision,
    mlpackage_filename,
)
from .pooling import MeanPoolingSentenceEmbedder

modernbert_mask_patch.apply()


@dataclass(frozen=True)
class ConversionRequest:
    huggingface_model_id: str
    output_basename: str
    sequence_length: int
    precision: ComputePrecision
    output_directory: Path = Path(".")


@dataclass(frozen=True)
class ConversionResult:
    output_path: Path
    trace_vs_eager_max_abs_diff: float
    coreml_vs_pytorch_cosine_similarity: float


def convert_to_coreml(request: ConversionRequest) -> ConversionResult:
    tokenizer = AutoTokenizer.from_pretrained(request.huggingface_model_id)
    encoder = AutoModel.from_pretrained(
        request.huggingface_model_id,
        attn_implementation=MODEL_ATTENTION_IMPLEMENTATION,
    )
    encoder.eval()
    embedder = MeanPoolingSentenceEmbedder(encoder).eval()

    example_input_ids, example_attention_mask = _build_example_encoding(tokenizer, request.sequence_length)

    with torch.no_grad():
        eager_output = embedder(example_input_ids, example_attention_mask)
        traced_embedder = torch.jit.trace(
            embedder, (example_input_ids, example_attention_mask), strict=False
        )
        traced_output = traced_embedder(example_input_ids, example_attention_mask)
    trace_max_diff = (eager_output - traced_output).abs().max().item()

    mlmodel = ct.convert(
        traced_embedder,
        inputs=[
            ct.TensorType(
                name=COREML_INPUT_IDS_FEATURE_NAME,
                shape=(EXPORT_BATCH_SIZE, request.sequence_length),
                dtype=int,
            ),
            ct.TensorType(
                name=COREML_ATTENTION_MASK_FEATURE_NAME,
                shape=(EXPORT_BATCH_SIZE, request.sequence_length),
                dtype=int,
            ),
        ],
        outputs=[ct.TensorType(name=COREML_OUTPUT_FEATURE_NAME)],
        convert_to="mlprogram",
        compute_precision=_resolve_compute_precision(request.precision),
        minimum_deployment_target=getattr(ct.target, MINIMUM_DEPLOYMENT_TARGET_NAME),
    )
    _annotate_metadata(mlmodel, request)

    output_path = request.output_directory / mlpackage_filename(
        request.output_basename, request.sequence_length, request.precision
    )
    mlmodel.save(str(output_path))

    verification_model = load_for_verification(output_path)
    coreml_output = verification_model.predict({
        COREML_INPUT_IDS_FEATURE_NAME: example_input_ids.numpy().astype(np.int32),
        COREML_ATTENTION_MASK_FEATURE_NAME: example_attention_mask.numpy().astype(np.int32),
    })[COREML_OUTPUT_FEATURE_NAME]

    return ConversionResult(
        output_path=output_path,
        trace_vs_eager_max_abs_diff=trace_max_diff,
        coreml_vs_pytorch_cosine_similarity=cosine_similarity(coreml_output, eager_output.numpy()),
    )


def _build_example_encoding(
    tokenizer: PreTrainedTokenizerBase, sequence_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        [VERIFICATION_PREFIX.value + VERIFICATION_TEXT],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
    )
    return encoded["input_ids"], encoded["attention_mask"]


def _resolve_compute_precision(precision: ComputePrecision) -> ct.precision:
    return ct.precision.FLOAT16 if precision is ComputePrecision.FLOAT16 else ct.precision.FLOAT32


def _annotate_metadata(mlmodel: ct.models.MLModel, request: ConversionRequest) -> None:
    mlmodel.author = f"Converted from {request.huggingface_model_id}"
    mlmodel.short_description = (
        f"Ruri Japanese sentence embedding ({request.huggingface_model_id}), "
        f"mean-pooled + L2-normalized, fixed sequence length {request.sequence_length}."
    )
    mlmodel.input_description[COREML_INPUT_IDS_FEATURE_NAME] = "Token ids, padded/truncated to a fixed length."
    mlmodel.input_description[COREML_ATTENTION_MASK_FEATURE_NAME] = "1 for real tokens, 0 for padding."
    mlmodel.output_description[COREML_OUTPUT_FEATURE_NAME] = "L2-normalized sentence embedding vector."
