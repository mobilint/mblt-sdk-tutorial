#!/usr/bin/env python3
"""Compile the exported SAM2 ONNX graphs into MBLT with ``mblt_compile``.

  python sam2_onnx_to_mblt.py

Step 2 of the encoder's two-step MBLT route: `sam2_export_onnx.py` writes the
ONNX, and this script parses it through qbcompiler's ONNX frontend, the same
``mblt_compile`` entry point the other tutorials in this repository use.

The decoder does not come this way. Its hypernetwork matmul is rejected by the
current parser, so it is produced by `sam2_decoder_to_mblt.py` through the
legacy parser instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET_DEVICE = "aries-rb"

# sam2_export_onnx.py names its output after --model-id; the tutorial's own
# artifacts use the shorter mxq-style stem. Both are accepted as defaults.
ONNX_STEMS = ("facebook_sam2-hiera-large", "sam2_hiera_large")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", choices=["encoder"], default="encoder")
    p.add_argument("--encoder-onnx", default=None, help="Encoder ONNX from sam2_export_onnx.py")
    p.add_argument("--encoder-mblt", default=None, help="Encoder MBLT output path")
    p.add_argument("--target-device", default=DEFAULT_TARGET_DEVICE, choices=["aries-rb"])
    p.add_argument("--dry-run", action="store_true", help="Report the resolved paths without compiling")
    return p.parse_args()


def find_onnx(part: str, override: str | None) -> Path:
    """Resolve the ONNX for `part`, accepting either naming convention."""
    if override:
        path = Path(override).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{part} ONNX not found: {path}")
        return path
    candidates = [SCRIPT_DIR / f"{stem}_{part}.onnx" for stem in ONNX_STEMS]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no {part} ONNX found. Looked for {[c.name for c in candidates]} in {SCRIPT_DIR}. "
        f"Run sam2_export_onnx.py --part {part} first, or pass --{part}-onnx."
    )


def compile_part(part: str, onnx_path: Path, mblt_path: Path, args: argparse.Namespace) -> None:
    from qbcompiler import mblt_compile

    # The encoder's shapes are fully static, so no example inputs are needed.
    print(f"[{part}] {onnx_path.name} -> {mblt_path.name}")
    mblt_compile(
        model=str(onnx_path),
        mblt_save_path=str(mblt_path),
        target_device=args.target_device,
        backend="onnx",
    )
    if not mblt_path.is_file():
        raise RuntimeError(f"mblt_compile did not create {mblt_path}")
    from decoder_bindings import read_mblt_input_names

    print(f"[{part}] wrote {mblt_path} ({mblt_path.stat().st_size} bytes)")
    print(f"[{part}] inputs: {read_mblt_input_names(mblt_path)}")


def main() -> None:
    args = parse_args()
    parts = [args.part]
    jobs = []
    for part in parts:
        onnx_path = find_onnx(part, getattr(args, f"{part}_onnx"))
        override = getattr(args, f"{part}_mblt")
        mblt_path = Path(override).resolve() if override else onnx_path.with_suffix(".mblt")
        jobs.append((part, onnx_path, mblt_path))
        print(f"  {part:7s} {onnx_path} -> {mblt_path}")
    if args.dry_run:
        raise SystemExit(0)

    failures = []
    for part, onnx_path, mblt_path in jobs:
        try:
            compile_part(part, onnx_path, mblt_path, args)
        except Exception as error:  # noqa: BLE001 - finish the other part, then report
            failures.append((part, error))
            print(f"[{part}] FAILED: {error}")
    if failures:
        raise SystemExit(f"{len(failures)} part(s) failed: {[part for part, _ in failures]}")


if __name__ == "__main__":
    main()
