"""Every named value the rest of the package depends on, in one place."""

from enum import Enum


class ComputePrecision(str, Enum):
    """The two Core ML compute precisions this pipeline supports.

    fp16 is the published format: half the size of fp32 and runs on the GPU
    and Neural Engine. fp32 is kept for debugging numerical differences.
    """

    FLOAT16 = "fp16"
    FLOAT32 = "fp32"


class RuriPromptPrefix(str, Enum):
    """ruri-v3's "1+3 prefix scheme": the text fed to the encoder should be
    prefixed according to the role it plays, matching how the base model was
    trained. See https://huggingface.co/cl-nagoya/ruri-v3-130m.
    """

    NONE = ""
    TOPIC = "トピック: "
    SEARCH_QUERY = "検索クエリ: "
    SEARCH_DOCUMENT = "検索文書: "


# --- Core ML I/O ------------------------------------------------------------

COREML_INPUT_IDS_FEATURE_NAME = "input_ids"
COREML_ATTENTION_MASK_FEATURE_NAME = "attention_mask"
COREML_OUTPUT_FEATURE_NAME = "sentence_embedding"

# Core ML input tensors carry an explicit batch dimension; this pipeline only
# ever traces/exports a single example at a time.
EXPORT_BATCH_SIZE = 1

# ModernBERT's HF implementation is traced with attn_implementation="eager"
# because SDPA's fused kernel isn't traceable by coremltools' torch frontend.
MODEL_ATTENTION_IMPLEMENTATION = "eager"

# The oldest Core ML deployment target the exported .mlpackage declares
# support for. iOS 16 is the floor for the ops this model's graph uses.
MINIMUM_DEPLOYMENT_TARGET_NAME = "iOS16"

# --- Pooling -----------------------------------------------------------------

# Guards the mean-pooling division against an all-padding input (an
# attention mask that sums to zero), which would otherwise divide by zero.
POOLING_DIVISION_EPSILON = 1e-9

L2_NORMALIZATION_DIMENSION = 1

# --- Verification -------------------------------------------------------------

# Loading a saved .mlpackage with the default (ALL) compute units has been
# observed to silently return NaN predictions on at least one macOS 27 beta,
# for reasons unrelated to the model itself (its GPU/ANE backend fails to
# validate and this appears to fall back incorrectly rather than raising).
# Verification's job is to check numerical correctness, not benchmark an
# accelerator, so it always forces CPU execution — guaranteed available and
# unaffected by that bug — regardless of what a shipped app later chooses.
VERIFICATION_COMPUTE_UNITS_NAME = "CPU_ONLY"

# A real ruri-v3 example ("Ruri (lapis lazuli) is a deep blue tinged with
# purple.") used to trace the model and to numerically sanity-check every
# conversion/quantization step's output against the original PyTorch model.
VERIFICATION_TEXT = "瑠璃色（るりいろ）は、紫みを帯びた濃い青のことである。"
VERIFICATION_PREFIX = RuriPromptPrefix.SEARCH_DOCUMENT

# --- Quantization --------------------------------------------------------------

# Below this many parameters, quantizing a weight tensor saves negligible
# space but still costs accuracy; coremltools skips such tensors when given
# this threshold.
INT8_QUANTIZATION_WEIGHT_THRESHOLD = 512
INT8_QUANTIZATION_MODE = "linear_symmetric"
INT8_QUANTIZATION_DTYPE = "int8"

# --- Naming --------------------------------------------------------------------


def mlpackage_filename(model_basename: str, sequence_length: int, precision: ComputePrecision) -> str:
    """The on-disk filename convention every script in this package uses,
    e.g. ``ruri-v3-130m_seq128_fp32.mlpackage``.
    """
    return f"{model_basename}_seq{sequence_length}_{precision.value}.mlpackage"
