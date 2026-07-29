# C++ Depth Estimation Runtime

This tutorial explains how to run a compiled `YOLO26m-depth` MXQ model with the C++ `qbruntime` API and save a colorized depth overlay.

Before starting, complete the compilation flow in [../../../compilation/depth_estimation/README.md](../../../compilation/depth_estimation/README.md). The examples below use `../../../compilation/depth_estimation/yolo26m-depth.mxq`.

## Prerequisites

Make sure the following components are available:

- Mobilint NPU driver and C++ `qbruntime`
- OpenCV development libraries
- A C++17 compiler
- CMake `3.21` or later
- The compiled `yolo26m-depth.mxq` model

For a native ARIES build on Ubuntu or Debian:

```bash
apt-get update
apt-get install -y build-essential cmake libopencv-dev
```

For REGULUS cross-compilation, activate the Mobilint toolchain described in the [C++ runtime guide](../README.md).

## Overview

The runtime flow in `infer_depth.cc` performs these steps:

1. Load and launch the MXQ model.
2. Read the input image and apply `768x768` YOLO-style letterbox preprocessing.
3. Pack a `uint8` RGB tensor in the HWC or CHW layout reported by the model.
4. Run inference on the Mobilint NPU.
5. Upsample the quarter-resolution MXQ depth output by 4×.
6. Remove letterbox padding and resize the depth map to the source image.
7. Colorize inverse depth and blend it over the source image.

The compilation tutorial fuses `/255` normalization into the MXQ model with `Uint8InputConfig`, so this runtime example supplies `uint8` pixels.

## Files in This Tutorial

- `infer_depth.cc`: Loads the model, runs inference, and saves the result.
- `utils/preprocess/`: Applies letterboxing, BGR-to-RGB conversion, and input-layout packing.
- `utils/postprocess/`: Restores the ONNX output shape, removes padding, and visualizes depth.
- `CMakeLists.txt`: Builds the `infer-depth` executable.

## Required MXQ Output Upsampling

The ONNX model returns `(1, 1, 768, 768)`, while the MXQ runtime returns a quarter-resolution `(1, 192, 192)` tensor. The C++ postprocessor performs the equivalent of:

```python
F.interpolate(
    depth,
    scale_factor=4.0,
    mode="bilinear",
    align_corners=False,
)
```

It uses OpenCV linear interpolation to resize `192x192` to `768x768`. OpenCV's half-pixel linear sampling matches PyTorch bilinear interpolation with `align_corners=False`. The program validates the 4× relationship before continuing.

After upsampling, the postprocessor removes the exact letterbox borders and restores the original image dimensions.

## Build

From this directory:

```bash
cmake -B build -S .
cmake --build build -j
```

This produces:

- `build/infer-depth`

You can inspect the target architecture with:

```bash
file build/infer-depth
```

## Run

The executable uses this interface:

```bash
./infer-depth <model.mxq> <image_path> <output_path>
```

### ARIES

Run the example with the shared bus image:

```bash
./build/infer-depth \
  ../../../compilation/depth_estimation/yolo26m-depth.mxq \
  ../../python/rc/bus.jpg \
  ./tmp/bus_depth_demo.jpg
```

### REGULUS

Copy `infer-depth`, `yolo26m-depth.mxq`, and `bus.jpg` to the target board, then run:

```bash
chmod +x infer-depth
./infer-depth yolo26m-depth.mxq bus.jpg bus_depth_demo.jpg
```

## Expected Output

The program prints the input shape, source image size, inference time, and raw MXQ output shape. It then saves a depth overlay such as `tmp/bus_depth_demo.jpg`, with nearer regions shown in warmer colors and farther regions in cooler colors.
