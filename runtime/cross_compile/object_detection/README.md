# Object Detection - C++ Cross-Compiled Inference

An example of running C++ NPU inference on a single image using a YOLOv9m MXQ model, with bounding box visualization.

## File Structure

- `infer_det.cc` - Inference binary source (NPU inference, post-processing, bbox visualization)
- `yolov9m_config.h` - YOLOv9m model configuration (preprocessing, postprocessing parameters)
- `utils/` - Shared inference modules (NPURunner, Transformer, YoloDecoder)
- `CMakeLists.txt` - CMake build configuration

## Prerequisites

- You need a `yolov9m.mxq` file generated with `model_compile_regulus.py` from the REGULUS section of the [compiler tutorial](../../../compilation/object_detection/README.md).

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
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE
make -j$(nproc)
cd ..
```

After a successful build, the `build/infer-det` binary will be generated.

Verify the binary is built for ARM64:

```bash
file build/infer-det
# build/infer-det: ELF 64-bit LSB executable, ARM aarch64, ...
```

## Run (Target Board)

Copy the `build/infer-det` binary and `yolov9m.mxq` model file to the target board, then run:

```bash
chmod +x infer-det
./infer-det yolov9m.mxq example.jpg result.jpg
```

## Example Output

```
Model input: 640x640x3
Image size: 1920x1080
Inference time: 15.234 ms
Detections: 3
  person 92% [120,45,380,520]
  car 87% [600,200,950,450]
  dog 76% [400,300,550,500]
Result saved to: result.jpg
```

The result image `result.jpg` contains the original image with bounding boxes and class labels drawn on it.
