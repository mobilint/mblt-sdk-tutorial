# C++ Semantic Segmentation Runtime

This tutorial explains how to run the compiled `YOLO26m-sem` MXQ model with the C++ `qbruntime` API and save a Cityscapes semantic-segmentation overlay.

Before starting, complete the [semantic-segmentation compilation tutorial](../../../compilation/semantic_segmentation/README.md). The examples below use `../../../compilation/semantic_segmentation/yolo26m-sem.mxq`.

## Prerequisites

Make sure the following components are available:

- Mobilint NPU driver and C++ `qbruntime`
- OpenCV development libraries
- A C++17 compiler
- CMake `3.21` or later
- The compiled `yolo26m-sem.mxq` model

For a native ARIES build on Ubuntu or Debian:

```bash
apt-get update
apt-get install -y build-essential cmake libopencv-dev
```

For REGULUS cross-compilation, activate the Mobilint toolchain described in the [C++ runtime guide](../README.md).

## Overview

The runtime flow in `infer_semantic.cc` performs these steps:

1. Load and launch the MXQ model.
2. Read `munster.png` and apply `1024x2048` YOLO-style letterbox preprocessing.
3. Pack a `uint8` RGB tensor in the HWC or CHW layout reported by the model.
4. Run inference on the Mobilint NPU.
5. Apply `argmax` across the 19 Cityscapes logits.
6. Remove letterbox padding and restore the class map to the source-image size.
7. Apply the official Cityscapes palette and blend it over the source image.

The compilation tutorial fuses `/255` normalization into the MXQ model with `Uint8InputConfig`, so this runtime example supplies `uint8` pixels.

## Files in This Tutorial

- `infer_semantic.cc`: Loads the model, runs inference, and saves the result.
- `utils/preprocess/`: Applies letterboxing, BGR-to-RGB conversion, and input-layout packing.
- `utils/postprocess/`: Applies `argmax`, restores the class map, and renders the Cityscapes overlay.
- `CMakeLists.txt`: Builds the `infer-semantic` executable.

## Input and Output Shapes

The compiled model reports:

- Input: `(1024, 2048, 3)` HWC `uint8`
- Output: `(1024, 2048, 19)` HWC `float32` logits

Unlike the ONNX model, whose graph includes `argmax` and returns a `(1, 1024, 2048)` class map, the MXQ model exposes all 19 logits. The C++ postprocessor computes the highest-scoring class for every pixel. It supports both HWC and CHW output layouts.

The MXQ spatial output size already matches the ONNX class-map size, so no output upsampling is required. Class maps are resized with nearest-neighbor interpolation only when restoring a differently sized source image.

The provided `munster.png` is already `2048x1024`, so the default model does not add letterbox padding.

## Build

From this directory:

```bash
cmake -B build -S .
cmake --build build -j
```

This produces:

- `build/infer-semantic`

You can inspect the target architecture with:

```bash
file build/infer-semantic
```

## Run

The executable uses this interface:

```bash
./infer-semantic <model.mxq> <image_path> <output_path>
```

### ARIES

Run the example with the shared Münster image:

```bash
./build/infer-semantic \
  ../../../compilation/semantic_segmentation/yolo26m-sem.mxq \
  ../../python/rc/munster.png \
  ./tmp/munster_semantic_demo.png
```

### REGULUS

Copy `infer-semantic`, `yolo26m-sem.mxq`, and `munster.png` to the target board, then run:

```bash
chmod +x infer-semantic
./infer-semantic yolo26m-sem.mxq munster.png munster_semantic_demo.png
```

## Expected Output

The program prints the input shape, source-image size, inference time, and raw MXQ output shape. It saves `tmp/munster_semantic_demo.png` at `2048x1024`, with each predicted class rendered in its official Cityscapes color.
