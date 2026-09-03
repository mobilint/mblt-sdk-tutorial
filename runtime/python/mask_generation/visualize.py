"""Mask overlay rendering for the SAM2 runtime tutorial."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MASK_COLORS = ((30, 144, 255), (255, 99, 71), (50, 205, 50))


def save_mask_overlays(
    image: np.ndarray,
    masks: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path,
) -> list[Path]:
    """Save one overlay per mask candidate with the prompt points drawn on top."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = np.asarray(image, dtype=np.uint8)
    saved: list[Path] = []
    for index, mask in enumerate(masks):
        color = np.asarray(MASK_COLORS[index % len(MASK_COLORS)], dtype=np.float32)
        canvas = base.astype(np.float32).copy()
        selected = np.asarray(mask) > 0
        canvas[selected] = 0.4 * canvas[selected] + 0.6 * color
        rendered = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(rendered)
        for (x, y), label in zip(points, labels):
            # Positive prompts are green, negative prompts are red.
            fill = "lime" if int(label) == 1 else "red"
            radius = 6
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
        path = output_dir / f"mask_{index}.png"
        rendered.save(path)
        saved.append(path)
    return saved
