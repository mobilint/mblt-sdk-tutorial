# Depth Estimation Runtime

This tutorial explains how to run a compiled `YOLO26m-depth` MXQ model with Mobilint `qbruntime` and save a colorized depth overlay.

Before starting, complete the compilation flow in [../../../compilation/depth_estimation/README.md](../../../compilation/depth_estimation/README.md). The default runtime command expects the compiled model at `../../../compilation/depth_estimation/yolo26m-depth.mxq`.

## Prerequisites

Make sure the following components are available:

- Mobilint NPU driver and `qbruntime`
- The compiled `yolo26m-depth.mxq` model
- Python packages: `opencv-python`, `numpy`, and `torch`

Install the additional Python packages if they are not already available:

```bash
pip install opencv-python numpy torch
```

See the [Python runtime guide](../README.md) for NPU driver and `qbruntime` setup.

## Overview

The runtime flow in `inference_mxq.py` performs these steps:

1. Load and launch the MXQ model with `qbruntime`.
2. Read an RGB image and apply `768x768` YOLO-style letterbox preprocessing.
3. Match the HWC or CHW input layout reported by the model.
4. Run inference on the Mobilint NPU.
5. Upsample the quarter-resolution MXQ output by 4× to match the ONNX output shape.
6. Remove letterbox padding and resize the depth map to the source image.
7. Colorize inverse depth and blend it over the source image.

The compiled MXQ model includes `/255` normalization, so the runtime input remains `uint8`.

## Files in This Tutorial

- `inference_mxq.py`: Runs preprocessing, NPU inference, postprocessing, and visualization.
- `postprocess.py`: Normalizes the MXQ output layout, upsamples it, and restores the source-image shape.
- `visualize.py`: Converts inverse depth to a JET color map and saves an overlay.

## Preprocessing

The script reads the model input shape and applies the same letterbox transform used during compilation. The default model expects `(768, 768, 3)` HWC input.

```python
model_input, borders = preprocess_yolo(image_rgb, input_shape)
outputs = model.infer([model_input])
```

The returned border sizes are passed to postprocessing so the padded regions can be removed accurately.

## Required MXQ Output Upsampling

The ONNX model returns a depth tensor with shape `(1, 1, 768, 768)`, while the compiled MXQ model returns `(1, 1, 192, 192)`. Therefore, the MXQ output must be bilinearly upsampled by a factor of four before letterbox restoration:

```python
depth = F.interpolate(
    depth,
    scale_factor=4.0,
    mode="bilinear",
    align_corners=False,
)
```

After this operation, `postprocess.py` verifies the `768x768` shape, removes the letterbox padding, and resizes the depth map to the original image dimensions.

## Run the Example

From this directory, run:

```bash
python inference_mxq.py
```

The default command uses:

- Model: `../../../compilation/depth_estimation/yolo26m-depth.mxq`
- Input image: `../rc/bus.jpg`
- Output image: `./tmp/bus_depth_demo.jpg`
- Depth overlay opacity: `0.7`

To specify paths and opacity explicitly:

```bash
python inference_mxq.py \
  --model-path ../../../compilation/depth_estimation/yolo26m-depth.mxq \
  --image-path ../rc/bus.jpg \
  --output-path ./tmp/bus_depth_demo.jpg \
  --overlay-alpha 0.7
```

## Parameters

- `--model-path`: Path to the compiled `.mxq` model.
- `--image-path`: Path to the input image.
- `--output-path`: Path for the visualized depth image.
- `--overlay-alpha`: Depth-map opacity from `0` to `1`. Default: `0.7`.

## Expected Output

The script saves `tmp/bus_depth_demo.jpg`. Nearer regions are shown with warmer colors and farther regions with cooler colors.
