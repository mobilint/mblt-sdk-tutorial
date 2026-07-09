# Face Detection Runtime in C++

This tutorial explains how to run a compiled face detection MXQ model with the C++ `qbruntime` API.

Before starting, complete the compilation flow in [../../../compilation/face_detection/README.md](../../../compilation/face_detection/README.md). The runtime example expects:

- `yolov12m-face.mxq`

## Prerequisites

Make sure the following components are available:

- Mobilint `qbruntime`
- OpenCV development libraries
- A C++17 compiler
- CMake `3.21` or later
- The compiled face-detection MXQ file

For ARIES native builds on Ubuntu or Debian:

```bash
apt-get update
apt-get install -y build-essential cmake libopencv-dev
```

For REGULUS cross-compilation, activate the Mobilint toolchain first as described in [../README.md](../README.md).

## Overview

The runtime flow in `infer_face.cc` follows these steps:

1. Load the compiled MXQ model.
2. Read the input image.
3. Apply YOLO-style letterbox preprocessing through `Transformer`.
4. Run inference on the Mobilint NPU.
5. Decode the single-class face detections with DFL and NMS.
6. Rescale detections back to the original image and draw the results.

This example uses a single-class `face` detector. It produces bounding boxes only; there are no landmark or keypoint outputs.

## Files in This Tutorial

- `infer_face.cc`: Runs the full face detection pipeline and saves the rendered image.
- `yolo_face_config.h`: Defines the face-detection head configuration, thresholds, and image size.
- `utils/inference/`: Shared runtime helpers for model execution and preprocessing.
- `utils/postprocess/`: Shared decode and NMS helpers.
- `CMakeLists.txt`: Builds the `infer-face` executable and supporting utility library.

## How the Program Works

The program uses this command-line interface:

```bash
./infer-face <model.mxq> <image_path> <output_path>
```

`Transformer` handles:

- Letterbox resize
- BGR-to-RGB conversion
- HWC-to-CHW conversion

After inference, `YoloDecoder` performs DFL decode, confidence filtering, NMS, and coordinate rescaling before the detections are drawn with the single `face` label.

## Build

From this directory:

```bash
cmake -B build -S .
cmake --build build -j
```

This produces:

- `build/infer-face`

You can verify the target architecture with:

```bash
file build/infer-face
```

## Run

Sample image:

- `../rc/cr7.jpg`

### ARIES

```bash
./build/infer-face ../../../compilation/face_detection/yolov12m-face.mxq ../rc/cr7.jpg result.jpg
```

### REGULUS

Copy `build/infer-face`, `yolov12m-face.mxq`, and `cr7.jpg` to the target board, then run:

```bash
chmod +x infer-face
./infer-face yolov12m-face.mxq cr7.jpg result.jpg
```

## Expected Output

The program prints the model input shape, original image size, inference time, and decoded detections, then saves an output image such as `result.jpg` with face bounding boxes and scores.
