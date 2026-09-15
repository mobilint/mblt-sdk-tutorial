#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import inspect
import random
import re
import tarfile
from pathlib import Path, PurePosixPath

from sav_dataset import VOS_FRAME_DIR, VOS_MASK_DIR, video_ids

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "sav"
DEFAULT_ARCHIVE = Path("./sav_val.tar")

# A video can produce fewer usable samples than requested, so each range allows
# for the worst case of one sample per video.
ENCODER_SAMPLES = 32
DECODER_SAMPLES = 60
ENCODER_VIDEOS_NEEDED = ENCODER_SAMPLES
DECODER_VIDEOS_NEEDED = DECODER_SAMPLES
VOS_MASK_RE = re.compile(rf"(?:^|/){VOS_MASK_DIR}/([^/]+)/([^/]+)/(\d+)\.png$")
VOS_FRAME_RE = re.compile(rf"(?:^|/){VOS_FRAME_DIR}/([^/]+)/(\d+)\.jpg$")
TRAIN_RE = re.compile(r"(?:^|/)([^/]+)_manual\.json$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract a calibration-ready subset from an SA-V archive")
    p.add_argument(
        "--archive",
        action="append",
        default=None,
        help=f"SA-V .tar to extract; repeatable. Default: {DEFAULT_ARCHIVE}",
    )
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to extract the subset. Default: data/sav next to this script",
    )
    p.add_argument(
        "--videos",
        type=int,
        default=120,
        help="Videos to keep. The default ranges are positional and need 100: encoder 0-31, "
        "decoder 36-95, with four-video gaps. Anything beyond that is the evaluation reserve.",
    )
    p.add_argument(
        "--frames-per-video",
        type=int,
        default=8,
        help="Annotated frames to keep per video. Calibration samples at most a few per video.",
    )
    p.add_argument("--seed", type=int, default=1234, help="Selection seed")
    p.add_argument("--dry-run", action="store_true", help="Report what would be extracted and exit")
    return p.parse_args()


def index_archive(archive: Path) -> tuple[str, dict]:
    """Scan archive headers and group members by video."""
    vos: dict[str, dict[str, list]] = collections.defaultdict(lambda: {"frames": {}, "masks": []})
    train: dict[str, list] = collections.defaultdict(list)
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            if match := VOS_MASK_RE.search(name):
                video, obj, frame = match.groups()
                vos[video]["masks"].append((int(frame), obj, member))
            elif match := VOS_FRAME_RE.search(name):
                video, frame = match.groups()
                vos[video]["frames"][int(frame)] = member
            elif match := TRAIN_RE.search(name):
                train[match.group(1)].append(member)
            elif name.endswith(".mp4"):
                train[Path(name).stem].append(member)
    if vos:
        return "vos", dict(vos)
    if train:
        complete = {
            video: members
            for video, members in train.items()
            if any(m.name.endswith(".mp4") for m in members) and any(m.name.endswith("_manual.json") for m in members)
        }
        return "train", complete
    raise ValueError(f"{archive.name}: no SA-V frames, masks, or videos found")


def select_vos(index: dict, videos: int, frames_per_video: int, seed: int) -> list:
    """Choose annotated frames and keep every object mask on each frame."""
    rng = random.Random(seed)
    chosen: list = []
    names = sorted(index)
    rng.shuffle(names)
    for video in names[:videos]:
        entry = index[video]
        by_frame: dict[int, list] = collections.defaultdict(list)
        for frame_index, _obj, member in entry["masks"]:
            if frame_index in entry["frames"]:
                by_frame[frame_index].append(member)
        if not by_frame:
            continue
        ordered = sorted(by_frame)
        if len(ordered) > frames_per_video:
            step = (len(ordered) - 1) / (frames_per_video - 1) if frames_per_video > 1 else 0
            ordered = [ordered[int(round(i * step))] for i in range(frames_per_video)]
            ordered = sorted(set(ordered))
        for frame_index in ordered:
            chosen.append(entry["frames"][frame_index])
            chosen.extend(by_frame[frame_index])
    return chosen


def select_train(index: dict, videos: int, seed: int) -> list:
    """Choose whole videos; the train layout cannot be subset below one mp4."""
    rng = random.Random(seed)
    names = sorted(index)
    rng.shuffle(names)
    chosen: list = []
    for video in names[:videos]:
        chosen.extend(m for m in index[video] if not m.name.endswith("_auto.json"))
    return chosen


RANGE_GAP = 4


def report(output_dir: Path, seed: int) -> None:
    """Print the disjoint calibration and evaluation ranges."""
    videos = len(video_ids(output_dir, seed=0))
    encoder_start = 0
    decoder_start = ENCODER_VIDEOS_NEEDED + RANGE_GAP
    decoder_end = decoder_start + DECODER_VIDEOS_NEEDED
    eval_start = decoder_end + RANGE_GAP

    print(f"\nSA-V root: {output_dir}")
    print(f"usable videos: {videos}")
    print("\ndisjoint video ranges (no video is shared between them):")
    print(f"  encoder calibration : {encoder_start:3d} - {ENCODER_VIDEOS_NEEDED - 1:3d}")
    print(f"  decoder calibration : {decoder_start:3d} - {decoder_end - 1:3d}")
    if videos > eval_start:
        print(f"  evaluation reserve  : {eval_start:3d} - {videos - 1:3d}  ({videos - eval_start} videos)")
    else:
        print(f"  evaluation reserve  : none; needs more than {eval_start} videos")
    if videos < decoder_end:
        print(
            f"\nwarning: only {videos} videos, but decoder calibration wants videos up to "
            f"{decoder_end - 1}. Lower sample counts while preserving nonoverlapping ranges, or prepare more videos."
        )
    print(
        f"\nNext:\n  python prepare_calibration.py --stage both \\\n"
        f"    --sav-root {output_dir} --seed {seed} \\\n"
        f"    --encoder-samples {ENCODER_SAMPLES} --encoder-skip-videos {encoder_start} "
        f"--encoder-max-videos {ENCODER_VIDEOS_NEEDED} \\\n"
        f"    --decoder-samples {DECODER_SAMPLES} --decoder-skip-videos {decoder_start} "
        f"--decoder-max-videos {DECODER_VIDEOS_NEEDED}"
    )


def safe_extract(tar: tarfile.TarFile, output_dir: Path, members: list[tarfile.TarInfo]) -> None:
    """Extract selected regular files and directories without traversal or links."""
    if "filter" in inspect.signature(tar.extractall).parameters:
        tar.extractall(output_dir, members=members, filter="data")
        return
    for member in members:
        path = PurePosixPath(member.name)
        destination = (output_dir / path).resolve(strict=False)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not destination.is_relative_to(output_dir.resolve())
            or not (member.isdir() or member.isreg())
        ):
            raise tarfile.TarError(f"unsafe archive member: {member.name}")
        tar.extract(member, path=output_dir)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    total = 0
    archives = args.archive or [str(DEFAULT_ARCHIVE)]
    for path in archives:
        archive = Path(path).resolve()
        if not archive.is_file():
            raise FileNotFoundError(f"--archive not found: {archive}")
        print(f"indexing {archive.name} ...", flush=True)
        layout, index = index_archive(archive)
        print(f"  layout: {layout}; videos in archive: {len(index)}")
        if layout == "vos":
            members = select_vos(index, args.videos, args.frames_per_video, args.seed)
        else:
            members = select_train(index, args.videos, args.seed)
        size = sum(m.size for m in members)
        print(f"  selected {len(members)} members ({size / 1e6:.0f} MB)")
        if args.dry_run:
            total += len(members)
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            safe_extract(tar, output_dir, members)
        total += len(members)
        print(f"  extracted into {output_dir}")

    if args.dry_run:
        print(f"\ndry run: {total} members would be extracted into {output_dir}")
        return
    report(output_dir, args.seed)


if __name__ == "__main__":
    main()
