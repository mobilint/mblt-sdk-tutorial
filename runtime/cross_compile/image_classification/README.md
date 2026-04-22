# Image Classification - C++ Cross-Compiled Inference

An example of running C++ NPU inference on a single image using a ResNet-50 MXQ model.

## File Structure

- `infer_cls.cc` - Inference binary source (NPU inference, Top-5 output)
- `CMakeLists.txt` - CMake build configuration
- `imagenet_labels.txt` - ImageNet 1000 class label file

## Prerequisites

- You need a `resnet50.mxq` file generated with `model_compile_regulus.py` from the Regulus section of the [compiler tutorial](../../../compilation/image_classification/README.md).

### 1. Verify Toolchain Installation

Check that the cross-compilation toolchain is installed:

```bash
ls /opt/crosstools/mobilint/
```

You should see a version directory (e.g., `1.0.0/`). If the toolchain is not installed, refer to [Cross-Compilation Setup](../README.md) to install it first.

### 2. Activate Cross-Compilation Environment

Find the `environment-setup-cortexa53-mobilint-linux` file in the installed toolchain directory:

```bash
find /opt/crosstools/mobilint/ -name "environment-setup-cortexa53-mobilint-linux"
# /opt/crosstools/mobilint/1.0.0/v3.4.0/environment-setup-cortexa53-mobilint-linux
```

Unset host library paths (e.g., CUDA) to prevent conflicts with the cross-compiler:

```bash
unset LD_LIBRARY_PATH
```

Activate the toolchain environment using the path found above:

```bash
source /opt/crosstools/mobilint/1.0.0/v3.4.0/environment-setup-cortexa53-mobilint-linux
```

Verify that the cross-compiler is set:

```bash
echo $CXX
# aarch64-mobilint-linux-g++ ...
```

## Build (Host)

```bash
mkdir build && cd build
cmake ..
make -j8
cd ..
```

After a successful build, the `build/infer-cls` binary will be generated.

Verify the binary is built for ARM64:

```bash
file build/infer-cls
# build/infer-cls: ELF 64-bit LSB executable, ARM aarch64, ...
```

## Run (Target Board)

Copy the `build/infer-cls` binary, `resnet50.mxq` model file, and `imagenet_labels.txt` label file to the target board, then run:

```bash
chmod +x infer-cls
./infer-cls resnet50.mxq example.jpg imagenet_labels.txt
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
