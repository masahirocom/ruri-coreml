#!/usr/bin/env bash
# Downloads the fp32 ruri-v3-130m Core ML model from Hugging Face into
# RuriDemo/Resources, where Xcode expects it. fp32 is used (not the smaller
# fp16/int8 variants also published) because fp16 has been observed to
# return NaN embeddings on real iPhones — see the top-level REPORT.md.
set -euo pipefail

REPO="masahiroid/ruri-v3-130m-coreml"
MODEL_NAME="ruri-v3-130m_seq128_fp32.mlpackage"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCES_DIR="$SCRIPT_DIR/../RuriDemo/Resources"
DEST="$RESOURCES_DIR/$MODEL_NAME"

if [ -d "$DEST" ]; then
  echo "Already present: $DEST"
  exit 0
fi

command -v huggingface-cli >/dev/null 2>&1 || {
  echo "huggingface-cli not found. Install it with: pip install huggingface_hub[cli]" >&2
  exit 1
}

mkdir -p "$RESOURCES_DIR"
echo "Downloading $MODEL_NAME from $REPO ..."
huggingface-cli download "$REPO" \
  --include "$MODEL_NAME/*" \
  --local-dir "$RESOURCES_DIR"

echo "Done -> $DEST"
