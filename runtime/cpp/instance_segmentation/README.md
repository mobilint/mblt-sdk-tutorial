# Instance Segmentation Runtime in C++

This tutorial explains how to run a compiled YOLO instance segmentation MXQ model with the C++ `qbruntime` API.

Before starting, complete the compilation flow in [../../../compilation/instance_segmentation/README.md](../../../compilation/instance_segmentation/README.md). The runtime example expects one of these compiled models:

- ARIES: `yolo11m-seg.mxq`
- REGULUS: `yolov8m-seg.mxq`

## Prerequisites

Make sure the following components are available:

- Mobilint `qbruntime`
- OpenCV development libraries
- A C++17 compiler
- CMake `3.21` or later
- The matching MXQ file from the compilation tutorial

For ARIES native builds on Ubuntu or Debian:

```bash
apt-get update
apt-get install -y build-essential cmake libopencv-dev
```

For REGULUS cross-compilation, activate the Mobilint toolchain first as described in [../README.md](../README.md).

## Overview

The runtime flow in `infer_seg.cc` follows these steps:

1. Load the compiled MXQ model.
2. Read the input image.
3. Apply YOLO-style letterbox preprocessing through `Transformer`.
4. Run inference on the Mobilint NPU.
5. Decode detections with DFL and NMS.
6. Assemble instance masks from the prototype tensor and mask coefficients.
7. Rescale detections back to the original image and draw masks, boxes, and labels.

The program uses `uint8` input and assumes normalization is fused into the compiled model.

## Files in This Tutorial

- `infer_seg.cc`: Runs the full instance segmentation pipeline and saves the rendered image.
- `yolo_seg_config.h`: Defines the segmentation-head configuration, thresholds, mask settings, and image size.
- `utils/inference/`: Shared runtime helpers for model execution and preprocessing.
- `utils/postprocess/`: Shared decode, mask assembly, and NMS helpers.
- `CMakeLists.txt`: Builds the `infer-seg` executable and supporting utility library.

## How the Program Works

The program uses this command-line interface:

```bash
./infer-seg <model.mxq> <image_path> <output_path>
```

`Transformer` handles:

- Letterbox resize
- BGR-to-RGB conversion
- HWC-to-CHW conversion

After inference, `YoloSegDecoder`:

- Decodes the anchor-free YOLO outputs
- Applies confidence filtering and NMS
- Extracts the prototype mask tensor
- Assembles one mask per detection
- Rescales the final detections to the original image

## Build

From this directory:

```bash
cmake -B build -S .
cmake --build build -j
```

This produces:

- `build/infer-seg`

You can verify the target architecture with:

```bash
file build/infer-seg
```

## Run

Sample image:

- `../rc/cr7.jpg`

### ARIES

```bash
./build/infer-seg ../../../compilation/instance_segmentation/yolo11m-seg.mxq ../rc/cr7.jpg result.jpg
```

### REGULUS

Copy `build/infer-seg`, `yolov8m-seg.mxq`, and `cr7.jpg` to the target board, then run:

```bash
chmod +x infer-seg
./infer-seg yolov8m-seg.mxq cr7.jpg result.jpg
```

## Expected Output

The program prints the model input shape, original image size, inference time, and decoded detections, then saves an output image such as `result.jpg` with instance masks, bounding boxes, and class labels.
