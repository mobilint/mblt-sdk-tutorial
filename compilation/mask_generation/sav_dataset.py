"""SA-V sampling helpers used to build SAM2 calibration data.

Encoder calibration only needs frames, while decoder calibration also needs a
ground-truth mask so a point prompt can be placed inside the object.

SA-V ships its splits in two different layouts, and both are supported here:

* **train** -- `{video}.mp4` beside `{video}_manual.json`, masks as RLE inside
  the json. This is what `sav_train` distributes.
* **vos** -- `JPEGImages_24fps/{video}/{frame}.jpg` beside
  `Annotations_6fps/{video}/{object}/{frame}.png`, one binary PNG per object
  per annotated frame. This is what `sav_val` and `sav_test` distribute.

`detect_layout` picks between them, so `--sav-root` accepts either and the
calibration scripts do not care which split the user obtained.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycocotools.mask as mask_util

# Mask-area fractions used to balance decoder calibration across object sizes.
AREA_BINS = ((0.0, 0.005), (0.005, 0.02), (0.02, 0.08), (0.08, 1.01))

# Directory names the SA-V val/test archives use.
VOS_FRAME_DIR = "JPEGImages_24fps"
VOS_MASK_DIR = "Annotations_6fps"


@dataclass(frozen=True)
class SavFrameSample:
    video: str
    frame_index: int
    frame: np.ndarray


@dataclass(frozen=True)
class SavMaskSample:
    video: str
    frame_index: int
    object_index: int
    frame: np.ndarray
    mask: np.ndarray


def decode_video_rgb(path: str | Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def decode_rle(value: dict[str, Any]) -> np.ndarray:
    mask = mask_util.decode(value)
    if mask.ndim == 3:
        mask = mask.sum(axis=2)
    return mask.astype(bool)


def detect_layout(sav_root: str | Path) -> str:
    """Return "train" for mp4 + `*_manual.json`, or "vos" for JPEG + PNG.

    The val/test archives extract to a `sav_val/`/`sav_test/` directory holding
    the two image directories, so the search is recursive and the layout is
    recognised whether `--sav-root` points at the extracted parent or at the
    split directory itself.
    """
    sav_root = Path(sav_root)
    if next(sav_root.rglob(VOS_FRAME_DIR), None) is not None:
        return "vos"
    return "train"


def vos_roots(sav_root: str | Path) -> list[Path]:
    """Return every directory holding the two VOS image directories.

    Extracting `sav_val.tar` and `sav_test.tar` into one output directory leaves
    two sibling split roots, so all of them are collected. Returning only the
    first would silently drop a whole split from calibration.
    """
    sav_root = Path(sav_root)
    if (sav_root / VOS_FRAME_DIR).is_dir():
        return [sav_root]
    roots = sorted({found.parent for found in sav_root.rglob(VOS_FRAME_DIR) if found.is_dir()})
    if not roots:
        raise FileNotFoundError(f"no {VOS_FRAME_DIR} directory under {sav_root}")
    return roots


def vos_mask_dir(video_dir: Path) -> Path:
    """Map `<root>/JPEGImages_24fps/<video>` to `<root>/Annotations_6fps/<video>`."""
    return video_dir.parent.parent / VOS_MASK_DIR / video_dir.name


def vos_video_dirs(sav_root: str | Path, seed: int) -> list[Path]:
    """List VOS videos that have both frames and annotations, shuffled by seed."""
    videos = [
        directory
        for root in vos_roots(sav_root)
        for directory in sorted((root / VOS_FRAME_DIR).iterdir())
        if directory.is_dir() and vos_mask_dir(directory).is_dir()
    ]
    random.Random(seed).shuffle(videos)
    return videos


def video_ids(sav_root: str | Path, seed: int = 0) -> list[str]:
    """List usable video ids for either layout, so callers can count them."""
    sav_root = Path(sav_root)
    if not sav_root.is_dir():
        return []
    if detect_layout(sav_root) == "vos":
        return [directory.name for directory in vos_video_dirs(sav_root, seed)]
    return [path.name.removesuffix("_manual.json") for path in annotation_pairs(sav_root, seed)]


def annotation_pairs(sav_root: str | Path, seed: int) -> list[Path]:
    """List `*_manual.json` annotations that have a matching video, shuffled by seed."""
    sav_root = Path(sav_root)
    pairs = []
    for annotation in sorted(sav_root.rglob("*_manual.json")):
        stem = annotation.name.removesuffix("_manual.json")
        if annotation.with_name(f"{stem}.mp4").is_file():
            pairs.append(annotation)
    random.Random(seed).shuffle(pairs)
    return pairs


def _distance_peak(mask: np.ndarray) -> tuple[float, float] | None:
    """Pick the most interior pixel of a mask as the first positive point."""
    distance = cv2.distanceTransform(mask.astype(np.uint8) * 255, cv2.DIST_L2, 5)
    if float(distance.max()) < 1.0:
        return None
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    return float(x), float(y)


def _second_positive(mask: np.ndarray, first: tuple[float, float], rng: random.Random) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    best: tuple[float, float] | None = None
    best_distance = -1.0
    for _ in range(64):
        index = rng.randint(0, len(xs) - 1)
        point = float(xs[index]), float(ys[index])
        distance = (point[0] - first[0]) ** 2 + (point[1] - first[1]) ** 2
        if distance > best_distance:
            best, best_distance = point, distance
    return best if best_distance > 9 else None


def _negative_point(mask: np.ndarray, height: int, width: int, rng: random.Random) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    dilated = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
    pad = max(8, int(0.15 * max(y1 - y0, x1 - x0)))
    bounds = (max(0, y0 - pad), min(height, y1 + pad), max(0, x0 - pad), min(width, x1 + pad))
    for global_search in (False, True):
        for _ in range(200):
            if global_search:
                x, y = rng.uniform(0, width - 1), rng.uniform(0, height - 1)
            else:
                y_min, y_max, x_min, x_max = bounds
                x, y = rng.uniform(x_min, x_max - 1), rng.uniform(y_min, y_max - 1)
            if not dilated[int(y), int(x)]:
                return x, y
    return None


def build_prompt(mask: np.ndarray, rng: random.Random, num_points: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Build a 1-, 2-, or 3-point prompt for one mask.

    One point is positive, two points add a negative point outside the mask,
    and three points add a second positive point far from the first.
    """
    height, width = mask.shape
    first = _distance_peak(mask)
    if first is None:
        return None
    if num_points == 1:
        return np.asarray([first], np.float32), np.asarray([1], np.int64)
    negative = _negative_point(mask, height, width, rng)
    if negative is None:
        return None
    if num_points == 2:
        return np.asarray([first, negative], np.float32), np.asarray([1, 0], np.int64)
    if num_points != 3:
        raise ValueError(f"num_points must be 1, 2, or 3; got {num_points}")
    second = _second_positive(mask, first, rng)
    if second is None:
        return None
    return (
        np.asarray([first, second, negative], np.float32),
        np.asarray([1, 1, 0], np.int64),
    )


def _area_bin(mask: np.ndarray) -> int:
    fraction = int(mask.sum()) / float(mask.size)
    return next(
        (index for index, (low, high) in enumerate(AREA_BINS) if low <= fraction < high),
        len(AREA_BINS) - 1,
    )


def _select_masks(
    masklet: list[Any],
    frames: list[np.ndarray],
    min_mask_area: int,
    per_video: int,
    rng: random.Random,
    bin_counts: list[int],
) -> list[tuple[int, int, np.ndarray]]:
    """Choose masks from one video, preferring the least-used area bin."""
    count = min(len(frames), len(masklet))
    if not count:
        return []
    positions = sorted({int(round(index * (count - 1) / 5)) for index in range(6)})
    candidates: list[tuple[int, int, np.ndarray, int]] = []
    for frame_index in positions:
        row = masklet[frame_index]
        if not isinstance(row, list):
            continue
        for object_index, rle in enumerate(row):
            if not isinstance(rle, dict) or "counts" not in rle:
                continue
            try:
                mask = decode_rle(rle)
            except Exception:
                continue
            if mask.shape != frames[frame_index].shape[:2] or int(mask.sum()) < min_mask_area:
                continue
            candidates.append((frame_index, object_index, mask, _area_bin(mask)))
    rng.shuffle(candidates)
    selected: list[tuple[int, int, np.ndarray]] = []
    used: set[tuple[int, int]] = set()
    while len(selected) < per_video:
        remaining = [item for item in candidates if item[:2] not in used]
        if not remaining:
            break
        target_bin = min({item[3] for item in remaining}, key=lambda value: bin_counts[value])
        item = next(value for value in remaining if value[3] == target_bin)
        used.add(item[:2])
        bin_counts[target_bin] += 1
        selected.append(item[:3])
    return selected


def _video_window(videos: list, skip_videos: int, max_videos: int | None) -> list:
    """Return the positional video range a calibration set is allowed to use.

    The bound belongs here rather than around the yielded samples: a video can
    produce nothing at all (every mask below ``--min-mask-area``, every prompt
    candidate rejected, an unreadable annotation), and such a video is invisible
    downstream. Counting only videos that yielded would let the set walk past its
    range into videos reserved for another set, which is the exact contamination
    the range exists to prevent. Slicing the list makes the range positional.
    """
    stop = None if max_videos is None else skip_videos + max_videos
    return videos[skip_videos:stop]


def _iter_frame_samples_train(
    sav_root: str | Path,
    *,
    seed: int,
    skip_videos: int,
    annotation_sample_rate: int,
    per_video: int,
    max_videos: int | None = None,
) -> Iterator[SavFrameSample]:
    """Yield evenly spaced frames per video for encoder calibration (train layout)."""
    rng = random.Random(seed + 11)
    for annotation in _video_window(annotation_pairs(sav_root, seed), skip_videos, max_videos):
        stem = annotation.name.removesuffix("_manual.json")
        frames = decode_video_rgb(annotation.with_name(f"{stem}.mp4"))[::annotation_sample_rate]
        if not frames:
            continue
        denominator = max(per_video - 1, 1)
        indices = {
            min(
                len(frames) - 1,
                max(0, int(round(i * (len(frames) - 1) / denominator)) + rng.randint(-1, 1)),
            )
            for i in range(per_video)
        }
        for frame_index in sorted(indices):
            yield SavFrameSample(stem, frame_index, np.ascontiguousarray(frames[frame_index]))


def _iter_mask_samples_train(
    sav_root: str | Path,
    *,
    seed: int,
    skip_videos: int,
    annotation_sample_rate: int,
    min_mask_area: int,
    per_video: int,
    max_videos: int | None = None,
) -> Iterator[SavMaskSample]:
    """Yield frame/mask pairs for decoder calibration (train layout)."""
    rng = random.Random(seed + 7)
    bins = [0] * len(AREA_BINS)
    for annotation in _video_window(annotation_pairs(sav_root, seed), skip_videos, max_videos):
        try:
            metadata = json.loads(annotation.read_text())
            masklet = metadata.get("masklet")
            if not isinstance(masklet, list) or not masklet:
                continue
            stem = annotation.name.removesuffix("_manual.json")
            frames = decode_video_rgb(annotation.with_name(f"{stem}.mp4"))[::annotation_sample_rate]
            if len(frames) != len(masklet):
                continue
            for frame_index, object_index, mask in _select_masks(masklet, frames, min_mask_area, per_video, rng, bins):
                yield SavMaskSample(
                    stem,
                    frame_index,
                    object_index,
                    np.ascontiguousarray(frames[frame_index]),
                    mask,
                )
        except Exception as error:
            print(f"skip {annotation}: {type(error).__name__}: {error}", flush=True)


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unreadable frame: {path}")
    return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _read_vos_mask(path: Path) -> np.ndarray | None:
    """Load one per-object PNG as a boolean mask.

    SA-V val/test store each object's mask as its own binary PNG with values in
    {0, 255}, so `> 0` binarizes it. A frame where the object is not visible is
    stored as an all-zero PNG -- roughly a third of them -- and is dropped here
    rather than becoming an unpromptable empty sample.
    """
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    mask = mask > 0
    return mask if mask.any() else None


def _iter_frame_samples_vos(
    sav_root: str | Path,
    *,
    seed: int,
    skip_videos: int,
    annotation_sample_rate: int,
    per_video: int,
    max_videos: int | None = None,
) -> Iterator[SavFrameSample]:
    """Yield evenly spaced frames per video for encoder calibration (vos layout)."""
    rng = random.Random(seed + 11)
    for video_dir in _video_window(vos_video_dirs(sav_root, seed), skip_videos, max_videos):
        frames = sorted(video_dir.glob("*.jpg"))[::annotation_sample_rate]
        if not frames:
            continue
        denominator = max(per_video - 1, 1)
        indices = {
            min(
                len(frames) - 1,
                max(0, int(round(i * (len(frames) - 1) / denominator)) + rng.randint(-1, 1)),
            )
            for i in range(per_video)
        }
        for frame_index in sorted(indices):
            frame_path = frames[frame_index]
            # Report the frame number encoded in the filename, not the position in
            # the stride-subsampled list, so a calibration sample recorded in
            # encoder_calib_samples.json can be traced back to its source frame.
            # The mask iterator already records `int(mask_path.stem)` for the same
            # reason, so both stages report comparable numbers.
            yield SavFrameSample(video_dir.name, int(frame_path.stem), _read_rgb(frame_path))


def _iter_mask_samples_vos(
    sav_root: str | Path,
    *,
    seed: int,
    skip_videos: int,
    annotation_sample_rate: int,
    min_mask_area: int,
    per_video: int,
    max_videos: int | None = None,
) -> Iterator[SavMaskSample]:
    """Yield frame/mask pairs for decoder calibration (vos layout).

    Mirrors the train path: candidates are spread over the video, then the
    least-used area bin is preferred so the calibration set stays balanced
    across object sizes rather than dominated by whichever size is common.
    """
    rng = random.Random(seed + 7)
    bins = [0] * len(AREA_BINS)
    for video_dir in _video_window(vos_video_dirs(sav_root, seed), skip_videos, max_videos):
        mask_dir = vos_mask_dir(video_dir)
        candidates: list[tuple[int, int, np.ndarray, int]] = []
        for object_dir in sorted(mask_dir.iterdir()):
            if not object_dir.is_dir():
                continue
            try:
                object_index = int(object_dir.name)
            except ValueError:
                continue
            masks = sorted(object_dir.glob("*.png"))[::annotation_sample_rate]
            if not masks:
                continue
            positions = sorted({int(round(i * (len(masks) - 1) / 5)) for i in range(6)})
            for position in positions:
                mask_path = masks[position]
                frame_path = video_dir / f"{mask_path.stem}.jpg"
                if not frame_path.is_file():
                    continue
                mask = _read_vos_mask(mask_path)
                if mask is None or int(mask.sum()) < min_mask_area:
                    continue
                candidates.append((int(mask_path.stem), object_index, mask, _area_bin(mask)))
        rng.shuffle(candidates)
        used: set[tuple[int, int]] = set()
        selected = 0
        while selected < per_video:
            remaining = [item for item in candidates if item[:2] not in used]
            if not remaining:
                break
            target_bin = min({item[3] for item in remaining}, key=lambda value: bins[value])
            frame_index, object_index, mask, bin_index = next(
                item for item in remaining if item[3] == target_bin
            )
            used.add((frame_index, object_index))
            bins[bin_index] += 1
            selected += 1
            frame = _read_rgb(video_dir / f"{frame_index:05d}.jpg")
            if mask.shape != frame.shape[:2]:
                continue
            yield SavMaskSample(video_dir.name, frame_index, object_index, frame, mask)



def iter_frame_samples(sav_root: str | Path, *, max_videos: int | None = None, **kwargs):
    """Yield encoder-calibration frames from whichever SA-V layout `sav_root` holds."""
    picker = _iter_frame_samples_vos if detect_layout(sav_root) == "vos" else _iter_frame_samples_train
    yield from picker(sav_root, max_videos=max_videos, **kwargs)


def iter_mask_samples(sav_root: str | Path, *, max_videos: int | None = None, **kwargs):
    """Yield decoder-calibration frame/mask pairs from whichever layout `sav_root` holds."""
    picker = _iter_mask_samples_vos if detect_layout(sav_root) == "vos" else _iter_mask_samples_train
    yield from picker(sav_root, max_videos=max_videos, **kwargs)
