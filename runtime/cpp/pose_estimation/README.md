# Pose Estimation Runtime in C++

This tutorial explains how to run a compiled YOLO pose-estimation MXQ model with the C++ `qbruntime` API.

Before starting, complete the compilation flow in [../../../compilation/pose_estimation/README.md](../../../compilation/pose_estimation/README.md). The runtime example expects one of these compiled models:

- REGULUS `regulus-rb` (default): `yolo11m-pose.mxq`
- REGULUS `regulus-ra`: `yolov8m-pose.mxq`
- ARIES `aries-rb`: `yolo11m-pose.mxq`

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

The runtime flow in `infer_pose.cc` follows these steps:

1. Load the compiled MXQ model.
2. Read the input image.
3. Apply YOLO-style letterbox preprocessing through `Preprocessor`.
4. Run inference on the Mobilint NPU.
5. Decode boxes and keypoints with DFL and NMS.
6. Rescale detections back to the original image.
7. Draw person boxes, keypoints, and skeleton limbs.

`--input-dtype` must match how the MXQ was compiled (see the [compilation tutorial](../../../compilation/pose_estimation/README.md)):

- `uint8`: MXQ compiled with fused normalization (`Uint8InputConfig`). Feeds raw letterboxed pixels.
- `float`: MXQ compiled without fusion. Applies `/255` at runtime.

If the flag does not match the compiled MXQ, the output is incorrect.

## Files in This Tutorial

- `infer_pose.cc`: Runs the full pose-estimation pipeline and saves the rendered image.
- `yolo_pose_config.h`: Defines the pose-head configuration, thresholds, keypoint count, and image size.
- `utils/preprocess/`: Preprocessing helpers (`Preprocessor`).
- `utils/postprocess/`: Shared decode and NMS helpers.
- `CMakeLists.txt`: Builds the `infer-pose` executable and supporting utility library.

## How the Program Works

The program uses this command-line interface:

```bash
./infer-pose <model.mxq> <image_path> <output_path> [--input-dtype uint8|float]
```

`Preprocessor` handles:

- Letterbox resize
- BGR-to-RGB conversion
- Packing into an HWC buffer for `Model::infer`

After inference, `YoloPoseDecoder`:

- Decodes the anchor-free YOLO outputs
- Extracts person boxes and keypoints
- Applies confidence filtering and NMS
- Rescales detections and keypoints to the original image

The visualizer then draws one-person boxes, 17 COCO keypoints, and the matching skeleton limbs.

## Build

From this directory:

```bash
cmake -B build -S .
cmake --build build -j
```

This produces:

- `build/infer-pose`

You can verify the target architecture with:

```bash
file build/infer-pose
```

## Run

Sample image:

- `../rc/cr7.jpg`

### ARIES

```bash
./build/infer-pose ../../../compilation/pose_estimation/yolo11m-pose.mxq ../rc/cr7.jpg result.jpg
```

### REGULUS (`regulus-rb`)

Copy `build/infer-pose`, `yolo11m-pose.mxq`, and `cr7.jpg` to the target board, then run:

```bash
chmod +x infer-pose
./infer-pose yolo11m-pose.mxq cr7.jpg result.jpg --input-dtype uint8   # MXQ compiled with fused normalization
./infer-pose yolo11m-pose.mxq cr7.jpg result.jpg --input-dtype float   # MXQ compiled without fusion
```

## Expected Output

The program prints the model input shape, original image size, inference time, and decoded detections, then saves an output image such as `result.jpg` with person boxes, keypoints, and skeleton lines.
