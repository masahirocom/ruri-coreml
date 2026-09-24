"""Prepares a local tokenizer folder an iOS app can bundle directly,
compatible with huggingface/swift-transformers' `AutoTokenizer.from(modelFolder:)`.

Why ``tokenizer_config.json`` is patched: ruri-v3's ``tokenizer_config.json``
declares ``tokenizer_class: "LlamaTokenizer"``, but swift-transformers maps
that class name to its BPE implementation. The actual tokenizer (declared in
``tokenizer.json``) is a Unigram/SentencePiece model, which has no merges
table and isn't compatible with a BPE parser — swift-transformers would
either crash or silently mis-tokenize. swift-transformers *does* map
``"XLMRobertaTokenizer"`` to its Unigram implementation, and XLM-RoBERTa's
tokenizer is architecturally the same kind of SentencePiece-Unigram model,
so declaring that class instead makes swift-transformers parse
``tokenizer.json`` correctly. Nothing about the actual vocabulary or
tokenization behavior changes — only which Swift class reads the file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

TOKENIZER_DATA_FILENAME = "tokenizer.json"
TOKENIZER_CONFIG_FILENAME = "tokenizer_config.json"

TOKENIZER_CLASS_CONFIG_KEY = "tokenizer_class"
IOS_COMPATIBLE_TOKENIZER_CLASS = "XLMRobertaTokenizer"

JSON_INDENT_WIDTH = 2


@dataclass(frozen=True)
class TokenizerExportResult:
    output_directory: Path
    original_tokenizer_class: str
    exported_tokenizer_class: str


def export_ios_tokenizer(huggingface_model_id: str, output_directory: Path) -> TokenizerExportResult:
    output_directory.mkdir(parents=True, exist_ok=True)

    tokenizer_data_path = Path(hf_hub_download(huggingface_model_id, TOKENIZER_DATA_FILENAME))
    (output_directory / TOKENIZER_DATA_FILENAME).write_bytes(tokenizer_data_path.read_bytes())

    tokenizer_config_path = Path(hf_hub_download(huggingface_model_id, TOKENIZER_CONFIG_FILENAME))
    config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    original_class = config.get(TOKENIZER_CLASS_CONFIG_KEY, "")
    config[TOKENIZER_CLASS_CONFIG_KEY] = IOS_COMPATIBLE_TOKENIZER_CLASS

    (output_directory / TOKENIZER_CONFIG_FILENAME).write_text(
        json.dumps(config, ensure_ascii=False, indent=JSON_INDENT_WIDTH),
        encoding="utf-8",
    )

    return TokenizerExportResult(
        output_directory=output_directory,
        original_tokenizer_class=original_class,
        exported_tokenizer_class=IOS_COMPATIBLE_TOKENIZER_CLASS,
    )
