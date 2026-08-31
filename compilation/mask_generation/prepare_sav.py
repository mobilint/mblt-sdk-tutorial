#!/usr/bin/env python3
"""Turn a SA-V archive you downloaded into a calibration-ready subset.

  python prepare_sav.py --archive sav_test.tar
  python prepare_sav.py --archive sav_val.tar --videos 40 --dry-run

Download SA-V yourself first. The official guide is
https://github.com/facebookresearch/sam2/blob/main/sav_dataset/README.md,
which points at the form-gated
https://ai.meta.com/datasets/segment-anything-video-downloads/ -- this script
never fetches anything, it only reads the `.tar` you already have.

Both SA-V layouts are accepted:

* `sav_val.tar` / `sav_test.tar` extract JPEG frames plus one binary PNG per
  object per annotated frame,
* `sav_train` chunks extract `{video}.mp4` beside `{video}_manual.json`.

Only a subset is extracted, because calibration needs a few hundred samples
rather than the whole split: a full `sav_val.tar` is 15 GB and 64,148 frames,
while the tutorial's defaults need 32 encoder and 300 decoder samples. Frames
without a matching annotation are skipped, so the extracted tree is a few
hundred MB instead of tens of GB.
"""

from __future__ import annotations

import argparse
import collections
import random
import re
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sav_dataset import VOS_FRAME_DIR, VOS_MASK_DIR, video_ids  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "sav"

# prepare_calibration.py defaults: skip 600 then 32/2 videos, and skip 800 then 300/4.
# Worst-case video budget, not samples/per_video: a video can yield fewer samples
# than requested (jittered frame indices collapsing, build_prompt rejecting thin
# masks), so a range sized by the arithmetic gets overrun. One sample per video is
# the floor, so N samples can need up to N videos.
ENCODER_SAMPLES = 32
DECODER_SAMPLES = 60
ENCODER_VIDEOS_NEEDED = ENCODER_SAMPLES
DECODER_VIDEOS_NEEDED = DECODER_SAMPLES
VOS_MASK_RE = re.compile(rf"(?:^|/){VOS_MASK_DIR}/([^/]+)/([^/]+)/(\d+)\.png$")
VOS_FRAME_RE = re.compile(rf"(?:^|/){VOS_FRAME_DIR}/([^/]+)/(\d+)\.jpg$")
TRAIN_RE = re.compile(r"(?:^|/)([^/]+)_manual\.json$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archive", action="append", required=True, help="SA-V .tar you downloaded; repeatable")
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to extract the subset. Default: data/sav next to this script",
    )
    p.add_argument(
        "--videos",
        type=int,
        default=120,
        help="Videos to keep. The defaults need 16 for the encoder and 75 for the decoder.",
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
    """Scan `archive` once and group its members by video.

    tarfile seeks past member data, so this reads only headers: a 15 GB
    `sav_val.tar` indexes in a couple of seconds.
    """
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
        # Keep only videos that have both the video and its manual annotation.
        complete = {
            video: members
            for video, members in train.items()
            if any(m.name.endswith(".mp4") for m in members)
            and any(m.name.endswith("_manual.json") for m in members)
        }
        return "train", complete
    raise ValueError(f"{archive.name}: no SA-V frames, masks, or videos found")


def select_vos(index: dict, videos: int, frames_per_video: int, seed: int) -> list:
    """Choose whole annotated frames, keeping every object mask on each frame.

    Selecting by frame rather than by mask means each extracted JPEG arrives
    with all of its masks, so decoder calibration can still balance across
    object sizes instead of being handed one arbitrary object per frame.
    """
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


# Videos left between ranges so a small change to sample counts does not overlap them.
RANGE_GAP = 4


def report(output_dir: Path, seed: int) -> None:
    """Print the video budget as three disjoint ranges over one split.

    Calibration and evaluation may come from the same split as long as no video
    is shared, which is what the skip offsets buy: the encoder takes the first
    videos, the decoder a later block, and whatever remains is reserved for
    evaluation. The shuffle is seeded, so these ranges are reproducible.
    """
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
            f"{decoder_end - 1}.\nLower --decoder-samples or --decoder-skip-videos, or prepare more videos."
        )
    # --*-max-videos turns each range into a hard bound. Without them the sets walk
    # past their range and quietly consume videos held back for evaluation.
    print(
        f"\nNext:\n  python prepare_calibration.py --stage both --defer-manifest \\\n"
        f"    --sav-root {output_dir} --seed {seed} \\\n"
        f"    --encoder-samples {ENCODER_SAMPLES} --encoder-skip-videos {encoder_start} "
        f"--encoder-max-videos {ENCODER_VIDEOS_NEEDED} \\\n"
        f"    --decoder-samples {DECODER_SAMPLES} --decoder-skip-videos {decoder_start} "
        f"--decoder-max-videos {DECODER_VIDEOS_NEEDED}"
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    total = 0
    for path in args.archive:
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
            # filter="data" rejects absolute paths, `..` traversal, and device nodes.
            tar.extractall(output_dir, members=members, filter="data")
        total += len(members)
        print(f"  extracted into {output_dir}")

    if args.dry_run:
        print(f"\ndry run: {total} members would be extracted into {output_dir}")
        return
    report(output_dir, args.seed)


if __name__ == "__main__":
    main()
