#!/usr/bin/env python3
"""Export the SAM2 Hiera image encoder to ONNX.

  python sam2_export_onnx.py
  python sam2_export_onnx.py --torch-device cpu

Step 1 of the encoder's two-step MBLT route. The input is captured from a real
`set_image` pass and `Sam2ImageEncoderWrapper` is traced to `.onnx`, which
`sam2_onnx_to_mblt.py` turns into `.mblt` through
`mblt_compile(..., backend="onnx")`.

This exports the encoder only. The decoder does not take this route: the
current parser rejects its hypernetwork matmul, so `sam2_decoder_to_mblt.py`
produces the decoder `.mblt` directly with the legacy parser, and no decoder
ONNX is written.

The qbcompiler patcher for the wrapper is applied before tracing, exactly as
the torch parser applies it, so the exported graph carries the device-friendly
rewrites (windowed attention without `unbind`, `LayerNorm2d` without `sqrt`,
`FpnNeck` upsampling through `nn.Upsample`) rather than the stock ones.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sam2_host as sam  # noqa: E402

DEFAULT_OPSET = 17

# Wrapper input order, fixed by Sam2MaskDecoderWrapper.forward.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default=sam.DEFAULT_MODEL_ID)
    p.add_argument("--base-path", default=".", help="Output directory.")
    p.add_argument("--part", choices=["encoder"], default="encoder")
    p.add_argument("--image", default=sam.DEFAULT_IMAGE)
    p.add_argument("--sam2-root", default=None, help="Local facebookresearch/sam2 checkout")
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version.")
    p.add_argument(
        "--torch-device",
        default=None,
        help="export device: cpu|cuda. Defaults to cuda when available, otherwise cpu. "
        "Tracing the Hiera-large encoder needs more than 12 GB of VRAM; use cpu on smaller cards.",
    )
    p.add_argument(
        "--patch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the qbcompiler SAM2 patcher before tracing. Exporting with --no-patch produces a stock "
        "graph that parses into more, smaller subgraphs.",
    )
    p.add_argument(
        "--fold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Constant-fold the exported graph with onnxruntime. Required for the encoder; see fold_constants().",
    )
    p.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the exported graph with onnxruntime and compare against the torch outputs.",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Tolerance for --verify, relative to each output's magnitude. onnxruntime runs on the CPU, so a "
        "CUDA export is compared across devices and a plain absolute tolerance would flag ordinary float noise.",
    )
    return p.parse_args()


def patcher_for(wrapper: torch.nn.Module):
    """Return the qbcompiler patcher context for `wrapper`, as the torch parser would."""
    from qbcompiler.model_dict_new.parser.patcher.models.hf_models.sam2 import get_patcher_cfg

    # get_patcher_cfg() binds the sam2 modeling modules the patchers reference, so it
    # must run before a patcher is instantiated.
    cfg = get_patcher_cfg()
    if not cfg.enabled:
        raise RuntimeError("qbcompiler could not bind the sam2 package; install sam2 or pass --sam2-root")
    entry = cfg.root_patcher_wrapper.get(type(wrapper))
    if entry is None:
        raise RuntimeError(f"no qbcompiler patcher registered for {type(wrapper).__name__}")
    patcher_cls = entry[0]
    return patcher_cls(wrapper)


def fold_constants(onnx_path: Path) -> None:
    """Constant-fold the exported graph in place with onnxruntime's BASIC optimizer.

    Hiera interpolates its position embedding to `x.shape`, which the TorchScript
    exporter records as a live `Shape`/`Gather`/`Div` chain feeding a `Resize`.
    qbcompiler cannot place that chain, so with `largest_supported_only` it cuts the
    graph there and the patch embed silently becomes a second graph input. The encoder
    input is fixed at 1024x1024, so every value in that chain is a constant; folding it
    away restores the single-input graph.

    BASIC and not EXTENDED: the extended level's fusions emit ops the ONNX frontend does
    not convert, and the parse then falls back to a 23-op fragment of the first block.
    """
    import onnxruntime as ort

    folded = onnx_path.with_suffix(".folded.onnx")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(folded)
    ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    folded.replace(onnx_path)


def export(wrapper, args_tuple, onnx_path: Path, input_names, output_names, opset, dynamic_axes=None, fold=True):
    """Trace `wrapper` to ONNX and report the graph interface."""
    torch.onnx.export(
        wrapper,
        args_tuple,
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        # The TorchScript exporter, not the dynamo one: qbcompiler's ONNX frontend
        # expects a classic opset graph.
        dynamo=False,
    )
    if fold:
        fold_constants(onnx_path)

    import onnx

    onnx.checker.check_model(str(onnx_path))
    model = onnx.load(str(onnx_path), load_external_data=False)
    print(f"write: {onnx_path} ({onnx_path.stat().st_size} bytes, opset {opset})")
    for tag, values in (("input", model.graph.input), ("output", model.graph.output)):
        for value in values:
            dims = [d.dim_param or d.dim_value for d in value.type.tensor_type.shape.dim]
            print(f"  {tag:6s} {value.name}: {dims}")


def as_numpy(value) -> np.ndarray:
    return np.ascontiguousarray(value.detach().float().cpu().numpy(), dtype=np.float32)


def verify(onnx_path: Path, feed: dict[str, np.ndarray], reference, names, tol: float) -> None:
    """Compare onnxruntime outputs against the torch outputs the export traced.

    The comparison is scaled by each output's own magnitude. SAM2 mask logits span
    roughly +/-16, so an absolute threshold tight enough for `iou_pred` rejects
    perfectly good masks purely on CUDA-versus-CPU float ordering.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, feed)
    reference = [reference] if isinstance(reference, torch.Tensor) else list(reference)
    if len(outputs) != len(reference):
        raise RuntimeError(f"{onnx_path.name}: onnx returned {len(outputs)} outputs, torch returned {len(reference)}")
    failed = []
    for name, got, want in zip(names, outputs, reference):
        want = as_numpy(want)
        if got.shape != want.shape:
            raise RuntimeError(f"{onnx_path.name}: output {name} shape {got.shape} != torch {want.shape}")
        absolute = float(np.abs(got.astype(np.float32) - want).max())
        scale = max(float(np.abs(want).max()), 1.0)
        relative = absolute / scale
        # A NaN anywhere makes `relative` NaN, and `relative > tol` is False for NaN,
        # so a numerically broken export would print MISMATCH and still exit 0. The
        # pass condition is therefore stated positively: anything not demonstrably
        # within tolerance fails.
        passed = bool(np.isfinite(relative)) and relative <= tol
        status = "ok" if passed else "MISMATCH"
        print(f"[verify] {onnx_path.name}: {name} abs {absolute:.3e} rel {relative:.3e} (tol {tol:g}) {status}")
        if not passed:
            failed.append(name)
    if failed:
        raise RuntimeError(f"{onnx_path.name}: exported graph does not match the torch outputs: {failed}")


def capture_encoder_input(predictor, image, torch_device: str) -> torch.Tensor:
    """Capture the preprocessed image the stock `set_image` feeds the encoder."""
    cap = {}
    handle = predictor.model.image_encoder.register_forward_pre_hook(lambda m, a: cap.__setitem__("img", a[0]))
    predictor.set_image(image)
    handle.remove()
    return cap["img"].to(torch_device).float()


def export_encoder(predictor, image, onnx_path: Path, args, torch_device: str) -> None:
    from qbcompiler.model_dict_new.parser.patcher.models.hf_models.sam2 import Sam2ImageEncoderWrapper

    input_image = capture_encoder_input(predictor, image, torch_device)
    print(f"[capture] encoder input shape: input_image={tuple(input_image.shape)}")

    wrapper = Sam2ImageEncoderWrapper(predictor.model).to(torch_device).eval()
    with patcher_for(wrapper) if args.patch else nullcontext():
        # The FPN level count comes from the neck config, so read it off a real forward
        # instead of hardcoding it. The same outputs are reused as the --verify reference,
        # because a second hiera-large forward is not cheap.
        with torch.no_grad():
            reference = wrapper(input_image)
        output_names = [f"backbone_fpn_{i}" for i in range(len(reference))]
        export(wrapper, (input_image,), onnx_path, ["input_image"], output_names, args.opset, fold=args.fold)
    if args.verify:
        verify(onnx_path, {"input_image": as_numpy(input_image)}, reference, output_names, args.tol)


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    # Match the compiled-artifact naming convention (`sam2_hiera_large_encoder.mxq`
    # from model_compile.py): drop the Hub namespace and normalize separators, so
    # the exported file is `sam2_hiera_large_encoder.onnx`.
    save_name = args.model_id.split("/")[-1].replace("-", "_").replace(".", "_")

    torch_device = sam.resolve_device(args.torch_device or "cuda")
    print(f"Using {torch_device.upper()} for ONNX export")
    # The --verify reference forward runs on this device while onnxruntime runs on the
    # CPU. Ampere-and-later GPUs use TF32 for conv/matmul by default, and its 10-bit
    # mantissa accumulates to ~2e-3 relative error across the Hiera-large trunk --
    # enough to fail an honest tolerance. Full fp32 keeps the comparison meaningful.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    predictor = sam.build_predictor(args.model_id, args.sam2_root, torch_device)
    image = sam.load_image_np(args.image)

    export_encoder(predictor, image, base_path / f"{save_name}_encoder.onnx", args, torch_device)


if __name__ == "__main__":
    main()
