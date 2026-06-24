# Oriented Bounding Boxes Runtime

This tutorial explains how to run the compiled `YOLO11m-obb` MXQ model with Mobilint `qbruntime`.

Before following this guide, complete the compilation step in [../../../compilation/oriented_bounding_boxes/README.md](../../../compilation/oriented_bounding_boxes/README.md). The runtime example expects the compiled model at `../../../compilation/oriented_bounding_boxes/yolo11m-obb.mxq`.

## Prerequisites

Make sure the following packages are available:

- `qbruntime`
- `opencv-python`
- `numpy`
- `torch`

Install the Python packages that are not already part of your SDK environment:

```bash
pip install opencv-python numpy torch
```

## Overview

The runtime pipeline in this directory performs five steps:

1. Load `yolo11m-obb.mxq` with `qbruntime`.
2. Apply the same `1024x1024` letterbox preprocessing used during compilation.
3. Run MXQ inference on the Mobilint runtime.
4. Decode DOTA-oriented bounding boxes and apply rotated NMS.
5. Render rotated polygons with class names and confidence scores.

The MXQ model already contains `/255` normalization through the compilation-time `Uint8InputConfig`, so this runtime example keeps the input as `uint8`.

## Files in This Tutorial

- `inference_mxq.py`: Runs MXQ inference and saves the rendered result image.
- `postprocess.py`: Decodes OBB outputs into `cx, cy, w, h, conf, cls, angle` rows.
- `visualize.py`: Rescales rotated boxes and draws polygons on the source image.
- `dota.py`: Defines DOTAv1 class names and a stable color palette.
- `utils.py`: Contains the minimal DFL, rotated-box decode, scaling, and rotated NMS helpers.

## Run the Example

Use the default model path, sample image, and output path:

```bash
python inference_mxq.py
```

This command uses:

- Model: `../../../compilation/oriented_bounding_boxes/yolo11m-obb.mxq`
- Input image: `../rc/airport.jpg`
- Output image: `./tmp/airport_demo.jpg`

You can also override the thresholds or file paths:

```bash
python inference_mxq.py --conf-thres 0.3 --iou-thres 0.5 --output-path ./tmp/airport_custom.jpg
```

## Notes

- This tutorial targets the `YOLO11m-obb` output layout specifically.
- The postprocess output consumed by `visualize.py` is a row tensor in the format `cx, cy, w, h, conf, cls, angle`.
- Full execution requires a valid Mobilint runtime environment and compatible hardware. If those are unavailable, you can still statically inspect and validate the scripts.
