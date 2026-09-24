#!/usr/bin/env bash
# Downloads ruri-v3 Core ML models from Hugging Face into
# RuriDemo/RuriDemo/Resources, where the Xcode project expects them.
set -euo pipefail

REPO="masahiroid/ruri-v3-130m-coreml"
MODEL_NAMES=(
  "ruri-v3-130m_seq128_fp16.mlpackage"
  "ruri-v3-130m_seq128_int8.mlpackage"
)
PACKAGE_FILES=(
  "Manifest.json"
  "Data/com.apple.CoreML/model.mlmodel"
  "Data/com.apple.CoreML/weights/weight.bin"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCES_DIR="$SCRIPT_DIR/../RuriDemo/Resources"
BASE_URL="https://huggingface.co/$REPO/resolve/main"

for model in "${MODEL_NAMES[@]}"; do
  dest="$RESOURCES_DIR/$model"
  if [ -f "$dest/Data/com.apple.CoreML/weights/weight.bin" ]; then
    echo "Already present: $dest"
    continue
  fi
  echo "Downloading $model from $REPO ..."
  for file in "${PACKAGE_FILES[@]}"; do
    curl -fL --create-dirs -o "$dest/$file" "$BASE_URL/$model/$file"
  done
done

echo "Done -> $RESOURCES_DIR"
