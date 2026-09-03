#!/usr/bin/env python3
"""Parse the SAM2 image encoder directly to ``.mblt``.

  python sam2_encoder_to_mblt.py

This is the direct torch-parser counterpart to ``sam2_decoder_to_mblt.py``.
It follows ``qbcompiler/scripts/sam2/sam2_devel_encoder.py``: capture the
tensor passed to ``SAM2Base.forward_image`` by a real ``set_image`` call, wrap
that method so its three FPN features are the graph outputs, and serialize the
legacy parser's model and weights as MBLT.  No ONNX file or ONNX frontend is
involved.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sam2_host as sam  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET_DEVICE = "aries-rb"


class Sam2ImageEncoderParsingWrapper(torch.nn.Module):
    """Expose SAM2's complete image path (trunk plus FPN neck) to the parser."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_image: torch.Tensor):
        return self.model.forward_image(input_image)["backbone_fpn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=sam.DEFAULT_MODEL_ID)
    parser.add_argument("--image", default=sam.DEFAULT_IMAGE)
    parser.add_argument("--sam2-root", default=None, help="Local facebookresearch/sam2 checkout")
    parser.add_argument("--target-device", default=DEFAULT_TARGET_DEVICE)
    parser.add_argument(
        "--save-path",
        default=str(SCRIPT_DIR / "sam2_hiera_large_encoder.mblt"),
        help="Output MBLT path. Default: next to this script",
    )
    parser.add_argument("--torch-device", default=None, help="Parse device: cpu|cuda. Defaults to cuda when available.")
    parser.add_argument("--ignore-weight", action="store_true", help="Skip weight serialization (structure check only)")
    return parser.parse_args()


def capture_encoder_input(predictor, image, device: torch.device) -> torch.Tensor:
    """Capture exactly what SAM2 supplies to ``forward_image`` during ``set_image``."""
    from qbcompiler.model_dict.parser.backend.hf.util import (
        DefaultInputsCaptureContainer,
        InputCaptureCtxManager,
    )

    captured = DefaultInputsCaptureContainer()
    with InputCaptureCtxManager(predictor.model, 1, captured, target_fn_name="forward_image"):
        predictor.set_image(image)
    return list(captured.captured_args[-1])[0].to(device)


def main() -> None:
    args = parse_args()
    from qbcompiler.model_dict.common import WeightDict
    from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor
    from qbcompiler.model_dict.parser.parser import ModelParser
    from qbcompiler.model_dict.serialize import ChainedByteObj, SerializeMeta

    torch_device = sam.resolve_device(args.torch_device or "cuda")
    print(f"Using {torch_device.upper()} for parsing")
    predictor = sam.build_predictor(args.model_id, args.sam2_root, torch_device)
    input_image = capture_encoder_input(predictor, sam.load_image_np(args.image), predictor.model.device)
    print(f"[capture] input_image={tuple(input_image.shape)}")

    parser = ModelParser(
        model=Sam2ImageEncoderParsingWrapper(predictor.model),
        backend="torch",
        target_device=args.target_device,
        yolo_decode_include=True,
    )
    parser.cfg.allocate_to_devices = True
    parser.cfg.split_supported_concat = True
    parser.cfg.operator_transform = True
    parser.parse(
        feed_dict={"input_image": wrap_tensor("input_image", input_image)},
        save_subgraph_type=1,
        debug=True,
    )

    model_dict = parser._model_dict
    weight_dict = WeightDict()
    for name, value in parser._weight_dict.items():
        weight_dict.add_weight(name, value)
    payload = SerializeMeta().serialize(model_dict, weight_dict, ignore_weight=args.ignore_weight)

    save_path = Path(args.save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as handle:
        payload.write(handle) if isinstance(payload, ChainedByteObj) else handle.write(payload)
    print(f"write: {save_path} ({save_path.stat().st_size} bytes)")
    for index, subgraph in enumerate(model_dict.subgraphs):
        print(f"  subgraph {index}: {len(subgraph.operators)} ops")


if __name__ == "__main__":
    main()
