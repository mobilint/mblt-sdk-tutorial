# Semantic Segmentation Runtime

This tutorial explains how to run the compiled `YOLO26m-sem` MXQ model with Mobilint `qbruntime` and save a Cityscapes semantic-segmentation overlay.

Before starting, complete the [semantic-segmentation compilation tutorial](../../../compilation/semantic_segmentation/README.md). The default command expects the compiled model at `../../../compilation/semantic_segmentation/yolo26m-sem.mxq`.

## Prerequisites

Make sure the following components are available:

- Mobilint NPU driver and `qbruntime`
- The compiled `yolo26m-sem.mxq` model
- Python packages: `opencv-python` and `numpy`

Install the additional Python packages if they are not already available:

```bash
pip install opencv-python numpy
```

Install `qbruntime` with the Python package or system package appropriate for your Mobilint environment. See the [Python runtime guide](../README.md) for driver and runtime setup.

## Overview

The runtime flow performs these steps:

1. Load and launch the MXQ model with `qbruntime`.
2. Read `munster.png` and apply `1024x2048` YOLO-style letterbox preprocessing.
3. Run inference on the Mobilint NPU.
4. Apply `argmax` across the 19 Cityscapes logits.
5. Remove letterbox padding and restore the class map to the source-image size.
6. Apply the Cityscapes palette and blend the result over the source image.

The compiled model includes `/255` normalization, so the runtime input remains `uint8`.

## Files in This Tutorial

- `inference_mxq.py`: Runs preprocessing, NPU inference, postprocessing, and visualization.
- `postprocess.py`: Applies `argmax` to the semantic logits and restores the source-image shape.
- `visualize.py`: Applies the Cityscapes palette and saves the overlay.

## Input and Output Shapes

The compiled model reports:

- Input: `(1024, 2048, 3)` HWC `uint8`
- Output: `(1024, 2048, 19)` HWC `float32` logits

Unlike the ONNX model, whose graph includes `argmax` and returns a `(1, 1024, 2048)` class map, the MXQ model exposes all 19 logits. The runtime must apply:

```python
class_map = np.argmax(logits, axis=-1)
```

The MXQ spatial output size already matches the ONNX class-map size, so no output upsampling is required.

## Preprocessing

`inference_mxq.py` reads the model input shape and applies the same centered letterbox transform used during compilation:

- Preserve the source aspect ratio.
- Resize with bilinear interpolation.
- Pad to `1024x2048` with RGB value `(114, 114, 114)`.
- Keep the input as `uint8`.

The provided `munster.png` is already `2048x1024`, so it does not require resizing or padding with the default model.

## Run the Example

From this directory, run:

```bash
python inference_mxq.py
```

The default command uses:

- Model: `../../../compilation/semantic_segmentation/yolo26m-sem.mxq`
- Input image: `../rc/munster.png`
- Output image: `./tmp/munster_semantic_demo.png`
- Overlay opacity: `0.7`

To specify all paths explicitly:

```bash
python inference_mxq.py \
  --model-path ../../../compilation/semantic_segmentation/yolo26m-sem.mxq \
  --image-path ../rc/munster.png \
  --output-path ./tmp/munster_semantic_demo.png \
  --overlay-alpha 0.7
```

## Parameters

- `--model-path`: Path to the compiled `.mxq` model.
- `--image-path`: Path to the input image.
- `--output-path`: Path for the visualized segmentation image.
- `--overlay-alpha`: Segmentation-overlay opacity from `0` to `1`. Default: `0.7`.

## Expected Output

The script saves `tmp/munster_semantic_demo.png` at `2048x1024`. Each predicted class uses its official Cityscapes color.
