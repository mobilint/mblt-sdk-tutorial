from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlretrieve

from ultralytics import YOLO

MODEL_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov12m-face.pt"
SCRIPT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = SCRIPT_DIR / os.path.basename(MODEL_URL)


def download_weights(weights_path: Path) -> Path:
    if weights_path.exists():
        print(f"Using existing weights: {weights_path}")
        return weights_path

    print(f"Downloading weights from {MODEL_URL}...")
    urlretrieve(MODEL_URL, weights_path)
    print(f"Saved weights to: {weights_path}")
    return weights_path


def export_onnx(weights_path: Path) -> Path:
    model = YOLO(weights_path)
    exported_path = Path(model.export(format="onnx"))
    print(f"Exported ONNX model to: {exported_path}")
    return exported_path


def main() -> None:
    weights_path = download_weights(WEIGHTS_PATH)
    export_onnx(weights_path)


if __name__ == "__main__":
    main()
