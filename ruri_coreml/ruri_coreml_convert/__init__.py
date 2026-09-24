"""Tools for exporting cl-nagoya/ruri-v3 (a Japanese ModernBERT-based
sentence embedding model) to Core ML for on-device use on iOS/macOS.

Each module has one job:

- ``constants``: every named value the rest of the package uses — sequence
  length choices, CoreML feature names, prefixes, quantization settings.
  Nothing outside this module should contain a literal that isn't obviously
  self-explanatory.
- ``modernbert_mask_patch``: works around a coremltools limitation when
  tracing ModernBERT's attention masking.
- ``pooling``: the mean-pooling + L2-normalize head wrapped around the base
  transformer.
- ``conversion``: PyTorch -> traced -> Core ML, plus a numerical sanity
  check against the original model.
- ``quantization``: int8 weight quantization of an already-converted model.
- ``tokenizer_export``: prepares a local tokenizer folder an iOS app can
  bundle, compatible with huggingface/swift-transformers.
- ``model_card``: renders the Hugging Face model card (README.md) shipped
  alongside a converted model.
- ``cli``: the `python -m ruri_coreml_convert` command-line entry point that
  ties the above together.
"""
