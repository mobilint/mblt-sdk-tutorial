"""Run point-prompted SAM2 segmentation with the compiled encoder and decoder MXQ models.

Pipeline:
    image -> SAM2 transform (host)
          -> encoder MXQ (NPU)
          -> prompt encoder and token assembly (host)
          -> decoder MXQ (NPU)
          -> mask upscaling and overlay (host)
"""

import json
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

import numpy as np
import qbruntime
import torch
from contracts import (
    DEFAULT_DECODER_RUNTIME_ORDER,
    build_decoder_runtime_feed,
    classify_decoder_outputs,
    strip_runtime_batch,
    validate_runtime_shapes,
)
from sam2_host import (
    build_predictor,
    fpn_from_runtime,
    install_runtime_features,
    load_rgb,
    postprocess_masks,
    prepare_decoder_tensors,
    preprocess_encoder_input,
)
from visualize import save_mask_overlays


def parse_point(value: str) -> tuple[float, float, int]:
    """Parse an `X,Y,LABEL` prompt in original image coordinates."""
    try:
        x, y, label = value.split(",")
        label_int = int(label)
        if label_int not in (0, 1):
            raise ValueError
        return float(x), float(y), label_int
    except ValueError as error:
        raise ArgumentTypeError(f"point must be X,Y,LABEL with LABEL 0 or 1; got {value!r}") from error


def launch_model(path: str, accelerator: qbruntime.Accelerator, core: qbruntime.Core) -> qbruntime.Model:
    """Launch one MXQ model pinned to a single core.

    The encoder and decoder are resident at the same time, so each is given its
    own core instead of letting both models claim the same one. The accelerator
    is passed in and kept alive by the caller for as long as both models run.
    """
    model_config = qbruntime.ModelConfig()
    model_config.set_single_core_mode(None, [qbruntime.CoreId(qbruntime.Cluster.Cluster0, core)])
    model = qbruntime.Model(path, model_config)
    model.launch(accelerator)
    return model


def get_torch_device() -> str:
    """Use CUDA when available and otherwise run the host SAM2 code on the CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    parser = ArgumentParser(description="Run SAM2 point-prompted segmentation with compiled MXQ models")
    parser.add_argument(
        "--encoder-mxq",
        type=str,
        default="../../../compilation/mask_generation/sam2_hiera_large_encoder.mxq",
        help="Path to the compiled encoder MXQ model",
    )
    parser.add_argument(
        "--decoder-mxq",
        type=str,
        default="../../../compilation/mask_generation/sam2_hiera_large_decoder.mxq",
        help="Path to the compiled decoder MXQ model",
    )
    parser.add_argument("--image-path", type=str, default="../rc/bus.jpg", help="Path to the input image")
    parser.add_argument(
        "--point",
        type=parse_point,
        action="append",
        required=True,
        metavar="X,Y,LABEL",
        help="Repeat for 1-3 positive(1) or negative(0) point prompts in original image coordinates",
    )
    parser.add_argument("--output-dir", type=str, default="./tmp/demo", help="Directory for overlays and outputs")
    parser.add_argument("--sam2-root", type=str, default=None, help="Local facebookresearch/sam2 checkout")
    parser.add_argument("--model-id", type=str, default="facebook/sam2-hiera-large", help="SAM2 model id")
    parser.add_argument(
        "--torch-device",
        type=str,
        default=None,
        help="Torch device for the host SAM2 code. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--decoder-runtime-order",
        type=str,
        default=",".join(DEFAULT_DECODER_RUNTIME_ORDER),
        help="Comma-separated semantic input order. For a rebuilt decoder read it from the "
        "calibration manifest's info['slot roles']; a shapes-only dump cannot tell the three "
        "(256, 64, 64) inputs apart",
    )
    args = parser.parse_args()

    if not 1 <= len(args.point) <= 3:
        raise ValueError(f"SAM2 decoder supports 1-3 point prompts; got {len(args.point)}")
    points = np.asarray([[point[0], point[1]] for point in args.point], dtype=np.float32)
    labels = np.asarray([point[2] for point in args.point], dtype=np.int64)

    image = load_rgb(args.image_path)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_device = args.torch_device or get_torch_device()
    print(f"Using {torch_device.upper()} for the host SAM2 model")
    predictor = build_predictor(args.model_id, args.sam2_root, torch_device)
    # One accelerator handle stays in scope until both models are disposed.
    accelerator = qbruntime.Accelerator()
    encoder = launch_model(args.encoder_mxq, accelerator, qbruntime.Core.Core0)
    decoder = launch_model(args.decoder_mxq, accelerator, qbruntime.Core.Core1)
    try:
        # Encoder: NHWC float32 [1, 1024, 1024, 3] with the batch axis stripped.
        encoder_feed = [strip_runtime_batch(preprocess_encoder_input(predictor, image))]
        validate_runtime_shapes(encoder_feed, encoder.get_model_input_shape(), "encoder")
        encoder_outputs = encoder.infer(encoder_feed)
        if encoder_outputs is None:
            raise RuntimeError("Encoder inference returned no outputs.")

        feature_maps = fpn_from_runtime(encoder_outputs, predictor.model.device)
        install_runtime_features(predictor, feature_maps, image.shape[:2])

        # Decoder: six inputs ordered by semantic role. The order matches the MBLT
        # input-name order, but roles keep the three same-shape inputs unambiguous.
        decoder_tensors = prepare_decoder_tensors(predictor, points, labels)
        decoder_feed = build_decoder_runtime_feed(decoder_tensors, args.decoder_runtime_order)
        validate_runtime_shapes(decoder_feed, decoder.get_model_input_shape(), "decoder")
        decoder_outputs = decoder.infer(decoder_feed)
        if decoder_outputs is None:
            raise RuntimeError("Decoder inference returned no outputs.")
    finally:
        decoder.dispose()
        encoder.dispose()

    result = classify_decoder_outputs(decoder_outputs)
    full_logits = postprocess_masks(predictor, result["masks"], image.shape[:2])
    binary_masks = full_logits > predictor.mask_threshold
    selected = int(np.argmax(result["iou"]))

    saved = {
        "masks": binary_masks,
        "full_logits": full_logits,
        "low_res_logits": result["masks"],
        "iou": result["iou"],
        "selected": np.asarray(selected, dtype=np.int64),
    }
    # Present only on decoders built with the four-output contract.
    for optional in ("sam_tokens", "object_score"):
        if optional in result:
            saved[optional] = result[optional]
    np.savez_compressed(output_dir / "outputs.npz", **saved)
    save_mask_overlays(image, binary_masks, points, labels, output_dir)

    summary = {
        "image": str(Path(args.image_path).resolve()),
        "points": points.tolist(),
        "labels": labels.tolist(),
        "selected": selected,
        "predicted_iou": np.asarray(result["iou"]).tolist(),
        "outputs": str(output_dir / "outputs.npz"),
    }
    if "object_score" in result:
        summary["object_score"] = np.asarray(result["object_score"]).tolist()
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
