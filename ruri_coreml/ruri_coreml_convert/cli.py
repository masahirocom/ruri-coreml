"""Command-line entry point: ``python -m ruri_coreml_convert <subcommand>``.

Each subcommand is a thin wrapper around one function from this package —
see that module for the actual logic. Run any subcommand with ``--help``
for its full argument list.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .constants import ComputePrecision
from .conversion import ConversionRequest, convert_to_coreml
from .model_card import ModelCardContext, write_model_card
from .quantization import QuantizationRequest, quantize_int8
from .tokenizer_export import export_ios_tokenizer

_DEFAULT_OUTPUT_DIRECTORY = Path(".")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ruri_coreml_convert")
    subparsers = parser.add_subparsers(required=True, dest="command")

    _add_convert_subcommand(subparsers)
    _add_quantize_subcommand(subparsers)
    _add_export_tokenizer_subcommand(subparsers)
    _add_model_card_subcommand(subparsers)

    return parser


def _add_convert_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("convert", help="Convert a ruri-v3 checkpoint to Core ML.")
    parser.add_argument("huggingface_model_id", help="e.g. cl-nagoya/ruri-v3-130m")
    parser.add_argument("output_basename", help="Filename prefix, e.g. ruri-v3-130m")
    parser.add_argument("sequence_length", type=int, help="Fixed token sequence length, e.g. 128")
    parser.add_argument(
        "--precision",
        type=ComputePrecision,
        choices=list(ComputePrecision),
        default=ComputePrecision.FLOAT32,
        metavar="{fp16,fp32}",
        help="Compute precision. fp32 is recommended for real-device use (see conversion.py).",
    )
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIRECTORY)
    parser.set_defaults(handler=_run_convert)


def _run_convert(args: argparse.Namespace) -> None:
    result = convert_to_coreml(ConversionRequest(
        huggingface_model_id=args.huggingface_model_id,
        output_basename=args.output_basename,
        sequence_length=args.sequence_length,
        precision=args.precision,
        output_directory=args.output_dir,
    ))
    print(f"saved -> {result.output_path}")
    print(f"trace max diff vs eager: {result.trace_vs_eager_max_abs_diff:.3e}")
    print(f"cosine(coreml, pytorch) = {result.coreml_vs_pytorch_cosine_similarity:.6f}")


def _add_quantize_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("quantize", help="int8-quantize an already-converted .mlpackage.")
    parser.add_argument("source_model_path", type=Path)
    parser.add_argument("destination_model_path", type=Path)
    parser.add_argument("tokenizer_huggingface_id", help="Model id to load a matching tokenizer from.")
    parser.add_argument("sequence_length", type=int)
    parser.set_defaults(handler=_run_quantize)


def _run_quantize(args: argparse.Namespace) -> None:
    result = quantize_int8(QuantizationRequest(
        source_model_path=args.source_model_path,
        destination_model_path=args.destination_model_path,
        tokenizer_huggingface_id=args.tokenizer_huggingface_id,
        sequence_length=args.sequence_length,
    ))
    print(f"saved -> {result.destination_path}")
    print(f"cosine(int8, source) = {result.quantized_vs_source_cosine_similarity:.6f}")


def _add_export_tokenizer_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "export-tokenizer",
        help="Prepare a local tokenizer folder for bundling into an iOS app.",
    )
    parser.add_argument("huggingface_model_id")
    parser.add_argument("output_directory", type=Path)
    parser.set_defaults(handler=_run_export_tokenizer)


def _run_export_tokenizer(args: argparse.Namespace) -> None:
    result = export_ios_tokenizer(args.huggingface_model_id, args.output_directory)
    print(f"wrote tokenizer to {result.output_directory}")
    print(f"tokenizer_class: {result.original_tokenizer_class} -> {result.exported_tokenizer_class}")


def _add_model_card_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("model-card", help="Render the Hugging Face README.md for a converted repo.")
    parser.add_argument("model_size_label", help="e.g. 130m")
    parser.add_argument("parameter_count_description", help="e.g. '132M params, hidden=512'")
    parser.add_argument("hidden_dimension", type=int)
    parser.add_argument("fp32_file_size_mb", type=int)
    parser.add_argument("fp16_file_size_mb", type=int)
    parser.add_argument("int8_file_size_mb", type=int)
    parser.add_argument("fp32_int8_file_size_mb", type=int)
    parser.add_argument("output_path", type=Path)
    parser.set_defaults(handler=_run_model_card)


def _run_model_card(args: argparse.Namespace) -> None:
    write_model_card(
        ModelCardContext(
            model_size_label=args.model_size_label,
            parameter_count_description=args.parameter_count_description,
            hidden_dimension=args.hidden_dimension,
            fp32_file_size_mb=args.fp32_file_size_mb,
            fp16_file_size_mb=args.fp16_file_size_mb,
            int8_file_size_mb=args.int8_file_size_mb,
            fp32_int8_file_size_mb=args.fp32_int8_file_size_mb,
        ),
        args.output_path,
    )
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
