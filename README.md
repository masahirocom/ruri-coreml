# ruri-coreml

[cl-nagoya/ruri-v3](https://huggingface.co/collections/cl-nagoya/ruri-japanese-general-text-embeddings) (a Japanese sentence embedding model) converted to Core ML for on-device use on iOS, plus an iOS app comparing it against Apple's own on-device embedding model (`NLContextualEmbedding`).

*(日本語版READMEはこちら: [README_ja.md](README_ja.md))*

**Requirements:** Xcode 16+, a physical iPhone (see `REPORT.md` for why the Simulator isn't recommended for this model), [XcodeGen](https://github.com/yonaskolb/XcodeGen), Python 3.10+ (only if converting a model yourself).

## What's here

- **[`RuriDemo/`](RuriDemo/)** — a SwiftUI iOS app that runs ruri-v3 via Core ML and Apple's `NLContextualEmbedding` side by side on the same queries, so you can compare their rankings directly on a real device.
- **[`ruri_coreml/ruri_coreml_convert/`](ruri_coreml/ruri_coreml_convert/)** — the Python pipeline used to convert ruri-v3 checkpoints to Core ML, quantize them, and prepare everything an iOS app needs to bundle one.
- **[`REPORT.md`](REPORT.md)** — a write-up comparing ruri-v3 against Apple's on-device Japanese/English embedding models on a semantic search task spanning modern Japanese, classical Japanese, and English.

## Converted models

Pre-converted `.mlpackage` files are published on Hugging Face — no need to run the conversion yourself unless you want a different size/sequence-length/precision:

- [masahiroid/ruri-v3-130m-coreml](https://huggingface.co/masahiroid/ruri-v3-130m-coreml)
- [masahiroid/ruri-v3-310m-coreml](https://huggingface.co/masahiroid/ruri-v3-310m-coreml)

**For real-device use, prefer the `fp32` variant.** fp16 models have been observed to return NaN embeddings on real iPhones (not the Simulator) — see `ruri_coreml/ruri_coreml_convert/conversion.py`'s module docstring and `REPORT.md` for details.

## Running the iOS app

```bash
cd RuriDemo
./scripts/download_model.sh   # fetches the fp32 model from Hugging Face into RuriDemo/Resources
xcodegen generate
open RuriDemo.xcodeproj
```

Build and run on a real device (the Simulator's CoreML backend has known issues with this model on some OS versions — see `REPORT.md`).

## Converting a model yourself

See [`ruri_coreml/README.md`](ruri_coreml/README.md).

## License

Apache License 2.0, matching the base [cl-nagoya/ruri-v3](https://huggingface.co/collections/cl-nagoya/ruri-japanese-general-text-embeddings) models. This repository only converts the model format; all credit for ruri-v3 itself belongs to its original authors (Nagoya University, cl-nagoya).
