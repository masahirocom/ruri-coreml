"""Workaround for a coremltools/transformers incompatibility when tracing
ModernBERT's attention masking.

``transformers.masking_utils.and_masks``/``or_masks`` build their combined
mask with ``Tensor.new_ones``/``new_zeros`` (see transformers' source). The
PyTorch->Core ML converter (coremltools' torch frontend) does not implement
either op and raises ``NotImplementedError`` when it encounters them. Both
functions are behaviorally trivial — an all-true (AND) or all-false (OR)
identity value that further mask terms get combined into — so this module
swaps in equivalents built from ``torch.ones``/``torch.zeros``, which the
converter does support, before tracing. It changes no math, only which
tensor-construction op reaches the traced graph.
"""

import torch
import transformers.masking_utils as _masking_utils

_IDENTITY_SHAPE = (1,)


def _traceable_and_masks(*mask_functions):
    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = torch.ones(_IDENTITY_SHAPE, dtype=torch.bool)
        for mask in mask_functions:
            result = result & mask(batch_idx, head_idx, q_idx, kv_idx)
        return result

    return and_mask


def _traceable_or_masks(*mask_functions):
    def or_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = torch.zeros(_IDENTITY_SHAPE, dtype=torch.bool)
        for mask in mask_functions:
            result = result | mask(batch_idx, head_idx, q_idx, kv_idx)
        return result

    return or_mask


def apply() -> None:
    """Monkey-patches ``transformers.masking_utils`` in place. Safe to call
    more than once. Must run before any ModernBERT model is constructed or
    traced, since transformers reads these functions from module scope at
    call time (not at import time), so a late patch still applies to models
    built beforehand — but calling this first keeps that from mattering.
    """
    _masking_utils.and_masks = _traceable_and_masks
    _masking_utils.or_masks = _traceable_or_masks
