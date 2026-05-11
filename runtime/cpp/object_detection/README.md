# Object Detection - C++ Inference (ARIES + REGULUS)

An example of running C++ NPU inference on a single image with bounding-box
visualization. The same binary handles both YOLO11m (ARIES) and YOLOv9m
(REGULUS) MXQ models because ultralytics v8/v9/v11 share the same anchor-free
Detect head layout (3 stride x [reg_max*4 box + nc cls] = 144 channels).

Supports **ARIES native build** (x86_64 host with NPU) and **REGULUS
cross-compile** (x86_64 host -> ARM64 target board) from the same
`CMakeLists.txt`.

## File Structure

- `infer_det.cc` - Inference binary source (NPU inference, post-processing, bbox visualization)
- `yolo_detect_config.h` - Anchorless YOLO detect configuration (yolo11m / yolov9m P5, 80 classes)
- `utils/` - Shared inference modules (NPURunner, Transformer, YoloDecoder)
- `CMakeLists.txt` - CMake build configuration (host arch auto-detected)

## Prerequisites

- Pick the matching MXQ file from the [compiler tutorial](../../../compilation/object_detection/README.md):
  - **ARIES**: `yolo11m.mxq` from `model_compile.py`.
  - **REGULUS**: `yolov9m.mxq` from `model_compile_regulus.py`.

### Common requirements (both paths)

- CMake >= 3.18
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

The same command works for both ARIES native and REGULUS cross-compile.
`CMakeLists.txt` detects the host arch and selects the right `-march` flag.

```bash
cmake -B build -S .
cmake --build build -j
```

After a successful build, `build/infer-det` is created.

Verify the architecture:

```bash
file build/infer-det
# ARIES:   ELF 64-bit LSB executable, x86-64, ...
# REGULUS: ELF 64-bit LSB executable, ARM aarch64, ...
```

## Run

A sample image `../rc/cr7.jpg` is bundled with the repo.

### ARIES (same host)

```bash
./build/infer-det ../../../compilation/object_detection/yolo11m.mxq ../rc/cr7.jpg result.jpg
```

### REGULUS (target board)

Copy `build/infer-det`, `yolov9m.mxq`, and `../rc/cr7.jpg` to the target board, then:

```bash
chmod +x infer-det
./infer-det yolov9m.mxq cr7.jpg result.jpg
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
