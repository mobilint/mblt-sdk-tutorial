# Image Classification - C++ Inference (ARIES + REGULUS)

An example of running C++ NPU inference on a single image using a ResNet-50 MXQ model.
Supports both **ARIES native build** (x86_64 host with NPU) and **REGULUS cross-compile**
(x86_64 host -> ARM64 target board) from the same `CMakeLists.txt`.

## File Structure

- `infer_cls.cc` - Inference binary source (NPU inference, Top-5 output)
- `CMakeLists.txt` - CMake build configuration (host arch auto-detected)
- `imagenet_labels.txt` - ImageNet 1000 class label file

## Prerequisites

- `resnet50.mxq` file generated with the matching compiler tutorial
  ([../../../compilation/image_classification/README.md](../../../compilation/image_classification/README.md)).
  ARIES uses `model_compile.py`, REGULUS uses `model_compile_regulus.py`.

### Common requirements (both paths)

- CMake >= 3.21
- C++17 compiler (gcc / clang)
- `qbruntime` library (installed together with the Mobilint NPU SDK)

### ARIES native build (x86_64 host with NPU)

Install host-side OpenCV and build tools (Ubuntu / Debian):

```bash
apt-get update
apt-get install -y build-essential cmake libopencv-dev
```

### REGULUS cross-compile (x86_64 host -> ARM64 target board)

The vendor cross-compile toolchain ships with OpenCV and `qbruntime` pre-installed.
Verify the toolchain and activate it:

```bash
ls /opt/crosstools/mobilint/                                   # version directory expected
unset LD_LIBRARY_PATH                                          # avoid host CUDA libs leaking
source /opt/crosstools/mobilint/{version}/{sdk}/environment-setup-cortexa53-mobilint-linux
echo $CXX                                                      # aarch64-mobilint-linux-g++ ...
```

If the toolchain is not installed, follow [Cross-Compilation Setup](../README.md).

## Build

The same command works for both ARIES native and REGULUS cross-compile. The
`CMakeLists.txt` detects the host arch and selects the right `-march` flag.

```bash
cmake -B build -S .
cmake --build build -j
```

After a successful build, `build/infer-cls` is created.

Verify the architecture:

```bash
file build/infer-cls
# ARIES:   ELF 64-bit LSB executable, x86-64, ...
# REGULUS: ELF 64-bit LSB executable, ARM aarch64, ...
```

## Run

A sample image `../rc/volcano.jpg` is bundled with the repo.

### ARIES (same host)

```bash
./build/infer-cls ../../../compilation/image_classification/resnet50.mxq ../rc/volcano.jpg imagenet_labels.txt
```

### REGULUS (target board)

Copy `build/infer-cls`, `resnet50.mxq`, `imagenet_labels.txt`, and `../rc/volcano.jpg` to the target board, then:

```bash
chmod +x infer-cls
./infer-cls resnet50.mxq volcano.jpg imagenet_labels.txt
```

## Example Output

```
Model input: 224x224x3
Inference time: 4.42336 ms

Top-5 predictions:
  980 volcano (12.345)
  970 alp (10.234)
  928 ice cream (8.567)
  949 strawberry (7.890)
  654 miniskirt (6.123)
```
