"""Generate SAM2 encoder and decoder calibration data from the SA-V dataset.

Encoder calibration is a plain list of preprocessed NHWC tensors. Decoder
calibration is a manifest that records the model input names, the semantic role
of each slot, and the per-slot tensor paths, because several decoder inputs
share the same shape.

The manifest is keyed by the input names the quantizer sees, which are the
POST-PARSE names: read from the decoder ``.mblt`` directly, or, for the ONNX
route, recovered by parsing the decoder ``.onnx`` the same way the compile
will (``--decoder-model`` accepts either). Decoder tensor generation does not
need the model at all, so ``--defer-manifest`` saves the tensors now and
``--stage manifest`` emits the manifest later, once a parseable decoder model
exists.
"""

import json
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from decoder_bindings import (
    load_binding_map,
    read_model_input_names,
    resolve_decoder_bindings,
)
from sam2_host import build_predictor, prepare_decoder_tensors, preprocess_encoder_input
from sav_dataset import build_prompt, detect_layout, iter_frame_samples, iter_mask_samples, video_ids

ENCODER_INPUT_SHAPE = (1, 1024, 1024, 3)

# Calibration tensors are written next to this script, not into the current working
# directory, so a run from anywhere still fills the tutorial's own calib/ tree.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENCODER_OUTPUT_DIR = SCRIPT_DIR / "calib" / "encoder"
DEFAULT_DECODER_OUTPUT_DIR = SCRIPT_DIR / "calib" / "decoder"


def parse_point_mix(values: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values.split(",") if value.strip())
    if not result or any(value not in (1, 2, 3) for value in result):
        raise ValueError("--point-mix must contain only 1, 2, or 3")
    return result


def generate_encoder_calibration(args, predictor) -> Path:
    """Save float32 NHWC encoder tensors and the listing file qbcompiler reads."""
    output_dir = Path(args.encoder_output_dir).resolve()
    tensor_dir = output_dir / "encoder"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sample in iter_frame_samples(
        args.sav_root,
        seed=args.seed,
        skip_videos=args.encoder_skip_videos,
        annotation_sample_rate=args.annotation_sample_rate,
        per_video=args.encoder_per_video,
        max_videos=args.encoder_max_videos,
    ):
        if len(records) >= args.encoder_samples:
            break
        tag = f"{len(records):05d}"
        value = preprocess_encoder_input(predictor, sample.frame)
        if value.shape != ENCODER_INPUT_SHAPE:
            raise ValueError(f"unexpected encoder calibration shape: {value.shape}")
        path = tensor_dir / f"{tag}.npy"
        np.save(path, value)
        records.append(
            {
                "tag": tag,
                "video": sample.video,
                "frame_index": sample.frame_index,
                "original_hw": list(sample.frame.shape[:2]),
                "path": str(path),
            }
        )
        print(f"[encoder {len(records)}/{args.encoder_samples}] {sample.video}:{sample.frame_index}")
    if len(records) != args.encoder_samples:
        raise RuntimeError(f"requested {args.encoder_samples} encoder samples, wrote {len(records)}")

    listing = output_dir / "encoder_calib.txt"
    listing.write_text("\n".join(record["path"] for record in records) + "\n")
    (output_dir / "encoder_calib_samples.json").write_text(
        json.dumps(
            {
                "num_samples": len(records),
                "seed": args.seed,
                "skip_videos": args.encoder_skip_videos,
                "per_video": args.encoder_per_video,
                "annotation_sample_rate": args.annotation_sample_rate,
                "sav_root": str(Path(args.sav_root).resolve()),
                "samples": records,
            },
            indent=2,
        )
        + "\n"
    )
    return listing


def read_decoder_input_names(args) -> tuple[Path, list[str]]:
    """Read the input names the decoder MBLT reports.

    Calibration must be keyed by the names the quantizer sees. Three decoder
    inputs share the shape ``(1, 256, 64, 64)``, so a positional guess would swap
    them silently; the binding map turns each name into a semantic role instead.
    """
    decoder_model = Path(args.decoder_model).resolve()
    if not decoder_model.is_file():
        raise FileNotFoundError(
            f"decoder model not found: {decoder_model}. Produce it with sam2_decoder_to_mblt.py."
        )
    try:
        return decoder_model, read_model_input_names(decoder_model)
    except Exception as error:
        raise RuntimeError(f"could not read the input contract from {decoder_model}: {error}") from error


def generate_decoder_tensors(args, predictor) -> Path:
    """Save the six decoder input tensors per sample plus the metadata the manifest needs.

    This half never touches the decoder model: the tensors are produced by the
    official FP32 host path and keyed by semantic role, so they can be generated
    before a parseable decoder model exists and reused for any input naming.
    """
    points_per_sample = parse_point_mix(args.point_mix)

    output_dir = Path(args.decoder_output_dir).resolve()
    tensor_root = output_dir / "decoder"

    rng = random.Random(args.seed + 17)
    records = []
    shapes_by_role: dict[str, list[int]] = {}
    for sample in iter_mask_samples(
        args.sav_root,
        seed=args.seed,
        skip_videos=args.decoder_skip_videos,
        annotation_sample_rate=args.annotation_sample_rate,
        min_mask_area=args.min_mask_area,
        per_video=args.decoder_per_video,
        max_videos=args.decoder_max_videos,
    ):
        if len(records) >= args.decoder_samples:
            break
        num_points = points_per_sample[len(records) % len(points_per_sample)]
        prompt = build_prompt(sample.mask, rng, num_points)
        if prompt is None:
            continue
        points, labels = prompt
        predictor.set_image(sample.frame)
        with torch.inference_mode():
            tensors = prepare_decoder_tensors(predictor, points, labels)
        for role in tensors:
            (tensor_root / role).mkdir(parents=True, exist_ok=True)
        # The prompt encoder emits one embedding per point plus one padding entry.
        # The 6 output tokens SAM2 prepends are concatenated inside the decoder graph
        # now, so they no longer appear in what the host hands over.
        expected_prompts = num_points + 1
        if tensors["sparse_prompt_embeddings"].shape != (1, 1, expected_prompts, 256):
            raise ValueError(
                f"unexpected sparse_prompt_embeddings for {num_points} points: "
                f"{tensors['sparse_prompt_embeddings'].shape}"
            )

        tag = f"{len(records):05d}"
        paths: dict[str, str] = {}
        for role, value in tensors.items():
            path = tensor_root / role / f"{tag}.npy"
            np.save(path, value)
            paths[role] = str(path)
            shapes_by_role.setdefault(role, list(value.shape))
        records.append(
            {
                "tag": tag,
                "video": sample.video,
                "frame_index": sample.frame_index,
                "object_index": sample.object_index,
                "mask_area": int(sample.mask.sum()),
                "num_points": num_points,
                "prompt_length": expected_prompts,
                "paths": paths,
            }
        )
        print(
            f"[decoder {len(records)}/{args.decoder_samples}] {sample.video}:{sample.frame_index} "
            f"points={num_points} prompts={expected_prompts}"
        )
    if len(records) != args.decoder_samples:
        raise RuntimeError(f"requested {args.decoder_samples} decoder samples, wrote {len(records)}")

    # Mark the prompt axis dynamic so the compiled decoder accepts 1-3 points.
    if len(set(points_per_sample)) > 1:
        shapes_by_role["sparse_prompt_embeddings"][2] = -1

    meta = output_dir / "decoder_tensor_meta.json"
    meta.write_text(
        json.dumps(
            {
                "source": "SA-V manual masklets",
                "num_samples": len(records),
                "seed": args.seed,
                "point_mix": list(points_per_sample),
                "shapes_by_role": shapes_by_role,
                "sample_paths_by_role": [record["paths"] for record in records],
            },
            indent=2,
        )
        + "\n"
    )
    (output_dir / "decoder_calib_samples.json").write_text(json.dumps(records, indent=2) + "\n")
    return meta


def write_decoder_manifest(args) -> Path:
    """Resolve the decoder input contract and emit the calibration manifest.

    Requires the tensors and metadata written by :func:`generate_decoder_tensors`.
    The token length fed to an ONNX contract parse comes from the recorded point
    mix, so the manifest step stays consistent with the tensors it describes.
    """
    output_dir = Path(args.decoder_output_dir).resolve()
    meta_path = output_dir / "decoder_tensor_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{meta_path} not found; generate decoder tensors first (--stage decoder)")
    meta = json.loads(meta_path.read_text())

    decoder_model, input_names = read_decoder_input_names(args)
    roles = resolve_decoder_bindings(input_names, load_binding_map(args.decoder_input_bindings))
    shapes_by_role = meta["shapes_by_role"]

    manifest = output_dir / "decoder_calib.json"
    manifest.write_text(
        json.dumps(
            {
                "info": {
                    "input names": input_names,
                    "slot roles": roles,
                    "input shapes": [shapes_by_role[role] for role in roles],
                    "source": meta["source"],
                    "num_samples": meta["num_samples"],
                    "seed": meta["seed"],
                    "point_mix": meta["point_mix"],
                    "decoder_model": str(decoder_model),
                },
                "calib paths": [[paths[role] for role in roles] for paths in meta["sample_paths_by_role"]],
            },
            indent=2,
        )
        + "\n"
    )
    return manifest


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate SAM2 encoder and decoder calibration data from SA-V")
    parser.add_argument(
        "--stage",
        choices=["encoder", "decoder", "both", "manifest"],
        default="both",
        help="Which calibration set to generate; `manifest` re-emits the decoder manifest from saved tensors",
    )
    parser.add_argument(
        "--sav-root",
        type=str,
        default=None,
        help="Path to the SA-V sav_train directory (not needed for --stage manifest)",
    )
    parser.add_argument("--sam2-root", type=str, default=None, help="Local facebookresearch/sam2 checkout")
    parser.add_argument("--model-id", type=str, default="facebook/sam2-hiera-large", help="SAM2 model id")
    parser.add_argument("--torch-device", type=str, default="cuda", help="Torch device for the host SAM2 model")
    parser.add_argument("--seed", type=int, default=1234, help="Shuffle seed for video selection")
    parser.add_argument("--annotation-sample-rate", type=int, default=4, help="Video frame stride")
    parser.add_argument("--min-mask-area", type=int, default=2000, help="Minimum mask area in pixels")

    parser.add_argument(
        "--encoder-output-dir",
        type=str,
        default=str(DEFAULT_ENCODER_OUTPUT_DIR),
        help="Encoder output directory. Default: calib/encoder next to this script",
    )
    parser.add_argument("--encoder-samples", type=int, default=32, help="Number of encoder samples")
    parser.add_argument("--encoder-skip-videos", type=int, default=600, help="Videos to skip for the encoder set")
    parser.add_argument("--encoder-per-video", type=int, default=2, help="Encoder frames per video")
    parser.add_argument(
        "--encoder-max-videos",
        type=int,
        default=None,
        help="Hard cap on videos the encoder set may span, keeping it inside its range",
    )

    parser.add_argument(
        "--decoder-output-dir",
        type=str,
        default=str(DEFAULT_DECODER_OUTPUT_DIR),
        help="Decoder output directory. Default: calib/decoder next to this script",
    )
    parser.add_argument("--decoder-samples", type=int, default=300, help="Number of decoder samples")
    parser.add_argument("--decoder-skip-videos", type=int, default=800, help="Videos to skip for the decoder set")
    parser.add_argument("--decoder-per-video", type=int, default=4, help="Decoder masks per video")
    parser.add_argument(
        "--decoder-max-videos",
        type=int,
        default=None,
        help="Hard cap on videos the decoder set may span, keeping it inside its range",
    )
    parser.add_argument("--point-mix", type=str, default="1,2,3", help="Point counts cycled across decoder samples")
    parser.add_argument(
        "--decoder-model",
        type=str,
        default="./sam2_hiera_large_decoder.mblt",
        help="Decoder model whose post-parse input names the manifest must match: a .mblt, or a .onnx "
        "from sam2_export_onnx.py (parsed the same way the compile will parse it)",
    )
    parser.add_argument(
        "--defer-manifest",
        action="store_true",
        help="Generate decoder tensors without emitting the manifest, for when no parseable decoder "
        "model exists yet; emit it later with --stage manifest",
    )
    parser.add_argument(
        "--decoder-input-bindings",
        type=str,
        default="./decoder_input_bindings.json",
        help="MBLT input name to semantic role map",
    )
    args = parser.parse_args()

    if args.stage != "manifest":
        if args.sav_root is None:
            parser.error("--sav-root is required unless --stage manifest")
        # Fail before the model load: scanning a missing or empty tree yields
        # nothing silently, which would surface only as "wrote 0 samples" at the end.
        if not Path(args.sav_root).is_dir():
            parser.error(f"--sav-root does not exist: {Path(args.sav_root).resolve()}")
        found = video_ids(args.sav_root, args.seed)
        if not found:
            parser.error(
                f"no SA-V videos under {Path(args.sav_root).resolve()}. Expected either the train layout "
                "(*_manual.json beside a matching .mp4) or the val/test layout (JPEGImages_24fps beside "
                "Annotations_6fps). Run prepare_sav.py on the archive you downloaded."
            )
        print(f"SA-V layout: {detect_layout(args.sav_root)} ({len(found)} videos)")

    if args.stage == "manifest":
        print(f"wrote {write_decoder_manifest(args)}")
        raise SystemExit(0)

    predictor = build_predictor(args.model_id, args.sam2_root, args.torch_device)
    if args.stage in ("encoder", "both"):
        print(f"wrote {generate_encoder_calibration(args, predictor)}")
    if args.stage in ("decoder", "both"):
        print(f"wrote {generate_decoder_tensors(args, predictor)}")
        if args.defer_manifest:
            print("manifest deferred; emit it later with --stage manifest --decoder-model <model>")
        else:
            print(f"wrote {write_decoder_manifest(args)}")
