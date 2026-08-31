#!/usr/bin/env python3
"""Parse the SAM2 mask decoder to `.mblt` with the legacy qbcompiler parser.

  python sam2_decoder_to_mblt.py

The ONNX route in Step 1 compiles the encoder but not the decoder: the decoder
dies in `mblt-graph`'s matmul transform with `unable to broadcast: 256, 32`.
That failure belongs to the *new* parser (`qbcompiler.model_dict_new`). The
legacy parser (`qbcompiler.model_dict`) lowers the same hypernetwork matmul
successfully, so the decoder can be produced today by routing it there.

Modelled on the reference at `qbcompiler/scripts/sam2/sam2_devel_decoder.py`.
Unlike the encoder route this parses `predictor.model.sam_mask_decoder`
directly rather than a wrapper: the parser captures the decoder's real call
arguments, then splits the graph itself into a host-side bridge (subgraph 0:
the output-token concat and the `image_embeddings + dense_prompt_embeddings`
sum) and the NPU body (subgraph 1). That is the same split the tutorial used to
perform by hand on the host before the decoder was parsed this way.

The prompt axis is marked dynamic on `sparse_prompt_embeddings`, so one decoder
serves any point count.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default=sam.DEFAULT_MODEL_ID)
    p.add_argument("--image", default=sam.DEFAULT_IMAGE)
    p.add_argument("--sam2-root", default=None, help="Local facebookresearch/sam2 checkout")
    p.add_argument("--target-device", default=DEFAULT_TARGET_DEVICE)
    p.add_argument(
        "--save-path",
        default=str(SCRIPT_DIR / "sam2_hiera_large_decoder.mblt"),
        help="Output MBLT path. Default: next to this script",
    )
    p.add_argument("--torch-device", default=None, help="parse device: cpu|cuda. Defaults to cuda when available.")
    p.add_argument("--ignore-weight", action="store_true", help="Skip weight serialization (structure check only)")
    return p.parse_args()


def capture_decoder_kwargs(predictor, image, points, labels):
    """Capture the mask decoder's real call arguments from a `predict` pass."""
    from qbcompiler.model_dict.parser.backend.hf.util import (
        DefaultInputsCaptureContainer,
        InputCaptureCtxManager,
    )

    predictor.set_image(image)
    container = DefaultInputsCaptureContainer()
    decoder = predictor.model.sam_mask_decoder
    with InputCaptureCtxManager(decoder, 1, container):
        predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)
    return decoder, container.captured_kwargs[-1]


def build_feed_dict(captured: dict, device) -> dict:
    """Wrap the captured tensors and mark the prompt axis dynamic.

    `sparse_prompt_embeddings` is `(1, N, 256)` for an N-point prompt, and it is
    what the in-graph token concat consumes. Marking axis -2 dynamic keeps one
    decoder usable for any point count instead of freezing it at the traced
    prompt, the same intent as the ONNX export's `dynamic_axes` on `tokens`.
    """
    from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor

    feed = {
        key: (wrap_tensor(key, value.to(device)) if isinstance(value, torch.Tensor) else value)
        for key, value in captured.items()
    }
    feed["high_res_features"] = [
        wrap_tensor(f"high_res_features{index}", value.to(device))
        for index, value in enumerate(captured["high_res_features"])
    ]
    feed["sparse_prompt_embeddings"].src_shape[-2].set_dynamic(True)
    return feed


def main() -> None:
    args = parse_args()
    from qbcompiler.model_dict.common import WeightDict
    from qbcompiler.model_dict.parser.parser import ModelParser
    from qbcompiler.model_dict.serialize import ChainedByteObj, SerializeMeta

    torch_device = sam.resolve_device(args.torch_device or "cuda")
    print(f"Using {torch_device.upper()} for parsing")
    predictor = sam.build_predictor(args.model_id, args.sam2_root, torch_device)
    points, labels = sam.prompt_arrays()
    decoder, captured = capture_decoder_kwargs(predictor, sam.load_image_np(args.image), points, labels)
    feed_dict = build_feed_dict(captured, predictor.model.device)

    parser = ModelParser(
        model=decoder,
        backend="torch",
        target_device=args.target_device,
        yolo_decode_include=True,
    )
    parser.cfg.allocate_to_devices = True
    parser.cfg.split_supported_concat = True
    parser.cfg.operator_transform = True
    # output_meta keeps (masks, iou_pred); the decoder also returns SAM tokens and
    # object-score logits, which the image path does not use.
    parser.parse(feed_dict=feed_dict, save_subgraph_type=1, debug=True, output_meta=lambda x: x[0][:2])

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
        role = "host bridge" if index == 0 else "NPU body"
        print(f"  subgraph {index} ({role}): {len(subgraph.operators)} ops")


if __name__ == "__main__":
    main()
