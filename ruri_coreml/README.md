# ruri_coreml_convert

Converts [cl-nagoya/ruri-v3](https://huggingface.co/collections/cl-nagoya/ruri-japanese-general-text-embeddings) (a Japanese ModernBERT-based sentence embedding model) to Core ML for on-device use on iOS/macOS, and prepares everything an app needs to bundle it: the `.mlpackage`, a local tokenizer folder, and a Hugging Face model card.

See [`RuriDemo/`](../RuriDemo) for an iOS app that consumes the converted model.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Convert a checkpoint to Core ML (fp16 by default).
python -m ruri_coreml_convert convert cl-nagoya/ruri-v3-130m ruri-v3-130m 128

# 2. Shrink it further with int8 weight quantization (recommended for mobile).
python -m ruri_coreml_convert quantize ruri-v3-130m_seq128_fp16.mlpackage ruri-v3-130m_seq128_int8.mlpackage cl-nagoya/ruri-v3-130m 128

# 3. Prepare a tokenizer folder an iOS app can bundle directly.
python -m ruri_coreml_convert export-tokenizer cl-nagoya/ruri-v3-130m ./tokenizer

# 4. Render a Hugging Face model card for a converted repo.
python -m ruri_coreml_convert model-card 130m "132M params, hidden=512" 512 252 127 README.md
```

Run any subcommand with `--help` for its full argument list.

## Package layout

Each module has exactly one job — see [`ruri_coreml_convert/__init__.py`](ruri_coreml_convert/__init__.py) for what each one is responsible for.

## Known issues this pipeline works around

- **coremltools can't trace ModernBERT's attention masking** — worked around in `modernbert_mask_patch.py`.
- **Loading a saved `.mlpackage` with default compute units can silently return NaN** on at least one macOS 27 beta — worked around in `_verification_loading.py` by forcing CPU-only for verification predictions.
