# Image Classification Runtime in C++

This tutorial explains how to run a compiled ResNet-50 MXQ model with the C++ `qbruntime` API.

Before starting, complete the compilation flow in [../../../compilation/image_classification/README.md](../../../compilation/image_classification/README.md). The runtime example uses a compiled `resnet50.mxq` model.

## Prerequisites

Make sure the following components are available:

- Mobilint `qbruntime`
- OpenCV development libraries
- A C++17 compiler
- CMake `3.21` or later
- `resnet50.mxq` from the matching compilation tutorial
- `imagenet_labels.txt` from this directory

For ARIES native builds on Ubuntu or Debian:

```bash
apt-get update
apt-get install -y build-essential cmake libopencv-dev
```

For REGULUS cross-compilation, activate the Mobilint toolchain first as described in [../README.md](../README.md).

## Overview

The runtime flow in `infer_cls.cc` follows these steps:

1. Load ImageNet labels from `imagenet_labels.txt`.
2. Load the compiled MXQ model with `qbruntime`.
3. Read the input image and apply ResNet-style preprocessing.
4. Run inference on the Mobilint NPU.
5. Print the top-5 class predictions.

`--input-dtype` must match how the MXQ was compiled (see the [compilation tutorial](../../../compilation/image_classification/README.md)):

- `uint8`: MXQ compiled with fused normalization (`Uint8InputConfig`). Feeds the cropped uint8 image directly.
- `float`: MXQ compiled without fusion. Applies `/255` and ResNet mean/std at runtime.

If the flag does not match the compiled MXQ, the output is incorrect.

## Files in This Tutorial

- `infer_cls.cc`: Runs the full image classification pipeline and prints top-5 predictions.
- `imagenet_labels.txt`: Label file for the 1000 ImageNet classes.
- `CMakeLists.txt`: Builds the `infer-cls` executable.

## How the Program Works

The program uses this command-line interface:

```bash
./infer-cls <model.mxq> <image_path> <labels_file> [--input-dtype uint8|float]
```

It preprocesses the input image by:

- Resizing the short edge to `256`
- Applying a `224x224` center crop
- Converting the image from BGR to RGB

After inference, it sorts the output logits and prints the top-5 class IDs, labels, and scores.

## Build

From this directory:

```bash
cmake -B build -S .
cmake --build build -j
```

This produces:

- `build/infer-cls`

You can verify the target architecture with:

```bash
file build/infer-cls
```

## Run

Sample image:

- `../rc/volcano.jpg`

### ARIES

```bash
./build/infer-cls ../../../compilation/image_classification/resnet50.mxq ../rc/volcano.jpg imagenet_labels.txt
```

### REGULUS (`regulus-rb`)

Copy `build/infer-cls`, `resnet50.mxq`, `imagenet_labels.txt`, and `volcano.jpg` to the target board, then run:

```bash
chmod +x infer-cls
./infer-cls resnet50.mxq volcano.jpg imagenet_labels.txt --input-dtype uint8   # MXQ compiled with fused normalization
./infer-cls resnet50.mxq volcano.jpg imagenet_labels.txt --input-dtype float   # MXQ compiled without fusion
```

## Expected Output

The program prints the model input shape, inference time, and top-5 ImageNet predictions.
