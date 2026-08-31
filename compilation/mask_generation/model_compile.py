"""Compile the SAM2 Hiera encoder and decoder MBLT models into MXQ.

Unlike the other compilation tutorials in this repository, SAM2 starts from
prebuilt `.mblt` graphs rather than an ONNX export, so this script calls
`mxq_compile` twice and never calls `mblt_compile`.
"""

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from decoder_bindings import load_binding_map, read_mblt_input_names, resolve_decoder_bindings

# Match prepare_calibration.py: the calibration tree lives next to these scripts,
# so the default calibration paths resolve the same way regardless of cwd.
SCRIPT_DIR = Path(__file__).resolve().parent


def get_compile_device() -> str:
    """Use CUDA when available and otherwise compile on the CPU."""
    return "gpu" if torch.cuda.is_available() else "cpu"


def validate_decoder_manifest(decoder_mblt: Path, manifest: Path, bindings: str | None) -> None:
    """Reject calibration that was generated against a different decoder MBLT.

    The decoder has several same-shape inputs, so a positional mismatch would
    quantize the wrong tensors without raising an error during compilation.
    """
    model_inputs = read_mblt_input_names(decoder_mblt)
    roles = resolve_decoder_bindings(model_inputs, load_binding_map(bindings))
    info = json.loads(manifest.read_text()).get("info", {})
    if info.get("input names") != model_inputs:
        raise ValueError(
            "decoder calibration input names do not match the MBLT. Regenerate calibration "
            "with this exact decoder MBLT instead of relying on positional same-shape inputs."
        )
    if info.get("slot roles") != roles:
        raise ValueError("decoder calibration slot roles do not match the binding map")


if __name__ == "__main__":
    parser = ArgumentParser(description="Compile SAM2 encoder and decoder MBLT models to MXQ")
    parser.add_argument(
        "--part",
        choices=["encoder", "decoder", "both"],
        default="both",
        help="Which model to compile. Use `encoder` while the decoder cannot be parsed.",
    )
    parser.add_argument(
        "--encoder-mblt",
        type=str,
        default="./sam2_hiera_large_encoder.mblt",
        help="Path to the SAM2 image encoder MBLT",
    )
    parser.add_argument(
        "--decoder-mblt",
        type=str,
        default="./sam2_hiera_large_decoder.mblt",
        help="Path to the SAM2 mask decoder MBLT",
    )
    parser.add_argument(
        "--encoder-calib",
        type=str,
        default=str(SCRIPT_DIR / "calib" / "encoder" / "encoder_calib.txt"),
        help="Encoder calibration listing produced by prepare_calibration.py",
    )
    parser.add_argument(
        "--decoder-calib",
        type=str,
        default=str(SCRIPT_DIR / "calib" / "decoder" / "decoder_calib.json"),
        help="Decoder calibration manifest produced by prepare_calibration.py",
    )
    parser.add_argument(
        "--encoder-save-path",
        type=str,
        default="./sam2_hiera_large_encoder.mxq",
        help="Path to save the encoder MXQ model",
    )
    parser.add_argument(
        "--decoder-save-path",
        type=str,
        default="./sam2_hiera_large_decoder.mxq",
        help="Path to save the decoder MXQ model",
    )
    parser.add_argument(
        "--compile-config",
        type=str,
        default="./compile_config.json",
        help="qbcompiler CompileConfig JSON",
    )
    parser.add_argument(
        "--decoder-input-bindings",
        type=str,
        default="./decoder_input_bindings.json",
        help="MBLT input name to semantic role map",
    )
    parser.add_argument(
        "--target-device",
        type=str,
        choices=["aries-rb"],
        default="aries-rb",
        help="Target NPU. Only aries-rb has been validated for SAM2.",
    )
    parser.add_argument("--inference-scheme", type=str, default="single", help="Inference scheme")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index used for calibration")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without compiling")
    args = parser.parse_args()

    # Before any CUDA query. torch.cuda.is_available() initializes and caches the
    # process's device visibility, so setting this afterwards would not remap it
    # and calibration could still land on GPU 0 or see every GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    parts = ["encoder", "decoder"] if args.part == "both" else [args.part]
    encoder_mblt = Path(args.encoder_mblt).resolve()
    decoder_mblt = Path(args.decoder_mblt).resolve()
    encoder_calib = Path(args.encoder_calib).resolve()
    decoder_calib = Path(args.decoder_calib).resolve()
    compile_config_path = Path(args.compile_config).resolve()
    required = [("compile-config", compile_config_path)]
    if "encoder" in parts:
        required += [("encoder-mblt", encoder_mblt), ("encoder-calib", encoder_calib)]
    if "decoder" in parts:
        required += [("decoder-mblt", decoder_mblt), ("decoder-calib", decoder_calib)]
    for label, path in required:
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")

    # Only the decoder has same-shape inputs that a positional mismatch could swap.
    if "decoder" in parts:
        validate_decoder_manifest(decoder_mblt, decoder_calib, args.decoder_input_bindings)

    compile_device = get_compile_device()
    print(f"Using {compile_device.upper()} for MXQ compilation")

    all_jobs = {
        "encoder": (encoder_mblt, encoder_calib, Path(args.encoder_save_path).resolve()),
        "decoder": (decoder_mblt, decoder_calib, Path(args.decoder_save_path).resolve()),
    }
    jobs = tuple((name, *all_jobs[name]) for name in parts)
    for name, model, calibration, save_path in jobs:
        print(f"  {name:7s} model={model}")
        print(f"          calib={calibration}")
        print(f"          output={save_path}")
    if args.dry_run:
        raise SystemExit(0)

    # Imported after CUDA_VISIBLE_DEVICES so calibration uses the requested GPU.
    from qbcompiler import mxq_compile
    from qbcompiler.configs import CompileConfig

    for name, model, calibration, save_path in jobs:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # The decoder MBLT carries two subgraphs: a host-side bridge that concatenates
        # the output tokens and adds the dense prompt, and the NPU body. Compiling a
        # multi-subgraph model requires cpu_offload, and its bridge mixes a CPU-resident
        # constant with the calibration tensors, so quantization runs on the CPU. The
        # encoder is a single subgraph and keeps the faster GPU path.
        is_multi_subgraph = name == "decoder"
        mxq_compile(
            model=str(model),
            target_device=args.target_device,
            calib_data_path=str(calibration),
            save_path=str(save_path),
            device="cpu" if is_multi_subgraph else compile_device,
            inference_scheme=args.inference_scheme,
            cpu_offload=is_multi_subgraph,
            compile_config=CompileConfig.from_file(str(compile_config_path)),
        )
        if not save_path.is_file():
            raise RuntimeError(f"mxq_compile did not create {save_path}")
        print(f"compiled {name}: {save_path} ({save_path.stat().st_size} bytes)")
