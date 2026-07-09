# Object Detection Runtime in C++

This tutorial explains how to run a compiled YOLO object detection MXQ model with the C++ `qbruntime` API.

Before starting, complete the compilation flow in [../../../compilation/object_detection/README.md](../../../compilation/object_detection/README.md). The runtime example expects one of these compiled models:

- ARIES: `yolo11m.mxq`
- REGULUS: `yolov9m.mxq`

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

The runtime flow in `infer_det.cc` follows these steps:

1. Load the compiled MXQ model.
2. Read the input image.
3. Apply YOLO-style letterbox preprocessing through `Transformer`.
4. Run inference on the Mobilint NPU.
5. Decode anchor-free YOLO outputs with DFL and NMS.
6. Rescale detections back to the original image and draw the results.

The program uses `uint8` input and assumes normalization is fused into the compiled model.

## Files in This Tutorial

- `infer_det.cc`: Runs the full object detection pipeline and saves the rendered image.
- `yolo_detect_config.h`: Defines the detect-head configuration, thresholds, and image size.
- `utils/inference/`: Shared runtime helpers for model execution and preprocessing.
- `utils/postprocess/`: Shared decode and NMS helpers.
- `CMakeLists.txt`: Builds the `infer-det` executable and supporting utility library.

## How the Program Works

The program uses this command-line interface:

```bash
./infer-det <model.mxq> <image_path> <output_path>
```

`Transformer` handles:

- Letterbox resize
- BGR-to-RGB conversion
- HWC-to-CHW conversion

After inference, `YoloDecoder` performs DFL decode, confidence filtering, NMS, and coordinate rescaling before the detections are drawn with COCO class labels.

## Build

From this directory:

```bash
cmake -B build -S .
cmake --build build -j
```

This produces:

- `build/infer-det`

You can verify the target architecture with:

```bash
file build/infer-det
```

## Run

Sample image:

- `../rc/cr7.jpg`

### ARIES

```bash
./build/infer-det ../../../compilation/object_detection/yolo11m.mxq ../rc/cr7.jpg result.jpg
```

### REGULUS

Copy `build/infer-det`, `yolov9m.mxq`, and `cr7.jpg` to the target board, then run:

```bash
chmod +x infer-det
./infer-det yolov9m.mxq cr7.jpg result.jpg
```

## Expected Output

The program prints the model input shape, original image size, inference time, and decoded detections, then saves an output image such as `result.jpg` with bounding boxes and class labels.
