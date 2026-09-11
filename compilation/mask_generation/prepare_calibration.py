import json
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from decoder_bindings import (
    load_binding_map,
    read_mblt_input_names,
    resolve_decoder_bindings,
)
from sam2_host import build_predictor, prepare_decoder_tensors, preprocess_encoder_input
from sav_dataset import build_prompt, detect_layout, iter_frame_samples, iter_mask_samples, video_ids

ENCODER_INPUT_SHAPE = (1, 1024, 1024, 3)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENCODER_OUTPUT_DIR = SCRIPT_DIR / "calib" / "encoder"
DEFAULT_DECODER_OUTPUT_DIR = SCRIPT_DIR / "calib" / "decoder"
DEFAULT_SAV_ROOT = SCRIPT_DIR / "data" / "sav"
# Disjoint ranges within prepare_sav.py's default 120-video subset.
DEFAULT_ENCODER_SKIP_VIDEOS = 0
DEFAULT_ENCODER_MAX_VIDEOS = 32
DEFAULT_DECODER_SKIP_VIDEOS = 36
DEFAULT_DECODER_MAX_VIDEOS = 60
DEFAULT_DECODER_SAMPLES = 60


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


def generate_decoder_tensors(args, predictor) -> Path:
    """Save six decoder tensors per sample and their semantic metadata."""
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
        expected_prompts = num_points + 1
        decoder = predictor.model.sam_mask_decoder
        output_tokens = decoder.num_mask_tokens + 1 + int(decoder.pred_obj_scores)
        expected_tokens = expected_prompts + output_tokens
        if tensors["tokens"].shape != (1, 1, expected_tokens, 256):
            raise ValueError(f"unexpected tokens for {num_points} points: {tensors['tokens'].shape}")

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
                "token_length": expected_tokens,
                "paths": paths,
            }
        )
        print(
            f"[decoder {len(records)}/{args.decoder_samples}] {sample.video}:{sample.frame_index} "
            f"points={num_points} tokens={expected_tokens}"
        )
    if len(records) != args.decoder_samples:
        raise RuntimeError(f"requested {args.decoder_samples} decoder samples, wrote {len(records)}")

    if len(set(points_per_sample)) > 1:
        shapes_by_role["tokens"][2] = -1

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


def write_decoder_manifest(
    decoder_model: str | Path,
    decoder_output_dir: str | Path = DEFAULT_DECODER_OUTPUT_DIR,
    decoder_input_bindings: str | Path | None = None,
) -> Path:
    """Write a calibration manifest in the generated MBLT input order."""
    output_dir = Path(decoder_output_dir).resolve()
    meta_path = output_dir / "decoder_tensor_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{meta_path} not found; generate decoder tensors first (--stage decoder)")
    meta = json.loads(meta_path.read_text())

    decoder_model = Path(decoder_model).resolve()
    if not decoder_model.is_file():
        raise FileNotFoundError(f"Decoder MBLT not found: {decoder_model}")
    input_names = read_mblt_input_names(decoder_model)
    roles = resolve_decoder_bindings(input_names, load_binding_map(decoder_input_bindings))
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
        choices=["encoder", "decoder", "both"],
        default="both",
        help="Which calibration set to generate",
    )
    parser.add_argument(
        "--sav-root",
        type=str,
        default=str(DEFAULT_SAV_ROOT),
        help="Extracted SA-V val/test or train root. Default: data/sav next to this script",
    )
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
    parser.add_argument(
        "--encoder-skip-videos",
        type=int,
        default=DEFAULT_ENCODER_SKIP_VIDEOS,
        help="Videos to skip for the encoder set",
    )
    parser.add_argument("--encoder-per-video", type=int, default=2, help="Encoder frames per video")
    parser.add_argument(
        "--encoder-max-videos",
        type=int,
        default=DEFAULT_ENCODER_MAX_VIDEOS,
        help="Hard cap on videos the encoder set may span, keeping it inside its range",
    )

    parser.add_argument(
        "--decoder-output-dir",
        type=str,
        default=str(DEFAULT_DECODER_OUTPUT_DIR),
        help="Decoder output directory. Default: calib/decoder next to this script",
    )
    parser.add_argument(
        "--decoder-samples", type=int, default=DEFAULT_DECODER_SAMPLES, help="Number of decoder samples"
    )
    parser.add_argument(
        "--decoder-skip-videos",
        type=int,
        default=DEFAULT_DECODER_SKIP_VIDEOS,
        help="Videos to skip for the decoder set",
    )
    parser.add_argument("--decoder-per-video", type=int, default=4, help="Decoder masks per video")
    parser.add_argument(
        "--decoder-max-videos",
        type=int,
        default=DEFAULT_DECODER_MAX_VIDEOS,
        help="Hard cap on videos the decoder set may span, keeping it inside its range",
    )
    parser.add_argument("--point-mix", type=str, default="1,2,3", help="Point counts cycled across decoder samples")
    args = parser.parse_args()

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

    if args.stage in ("encoder", "both"):
        (Path(args.encoder_output_dir) / "encoder_calib.txt").unlink(missing_ok=True)
    if args.stage in ("decoder", "both"):
        decoder_output_dir = Path(args.decoder_output_dir)
        (decoder_output_dir / "decoder_calib.json").unlink(missing_ok=True)
        (decoder_output_dir / "decoder_tensor_meta.json").unlink(missing_ok=True)

    predictor = build_predictor(args.model_id, args.torch_device)
    if args.stage in ("encoder", "both"):
        print(f"wrote {generate_encoder_calibration(args, predictor)}")
    if args.stage in ("decoder", "both"):
        print(f"wrote {generate_decoder_tensors(args, predictor)}")
