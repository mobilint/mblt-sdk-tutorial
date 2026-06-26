# Instance Segmentation - C++ Inference (ARIES + REGULUS)

An example of running C++ NPU inference on a single image with instance-mask and
bounding-box visualization. The same binary handles the YOLO11m-seg MXQ model on
both ARIES and REGULUS because the ultralytics v8/v11 anchor-free Segment head
shares the same output layout (3 stride x [reg_max*4 box + nc cls + 32 mask] channels,
plus one prototype tensor of shape [32, 160, 160]).

Supports **ARIES native build** (x86_64 host with NPU) and **REGULUS
cross-compile** (x86_64 host -> ARM64 target board) from the same
`CMakeLists.txt`.

## File Structure

- `infer_seg.cc` - Inference binary source (NPU inference, post-processing, mask + bbox visualization)
- `yolo_seg_config.h` - Anchorless YOLO segment configuration (yolo11m-seg / yolov8m-seg P5, 80 classes, 32 mask coefficients)
- `utils/` - Shared inference modules (NPURunner, Transformer, YoloSegDecoder)
- `CMakeLists.txt` - CMake build configuration (host arch auto-detected)

## Prerequisites

- Pick the matching MXQ file from the [compiler tutorial](../../../compilation/instance_segmentation/README.md):
  - **ARIES**: `yolo11m-seg.mxq` from `model_compile.py`.
  - **REGULUS**: `yolov8m-seg.mxq` from `model_compile_regulus.py`.

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

The same command works for both ARIES native and REGULUS cross-compile.
`CMakeLists.txt` detects the host arch and selects the right `-march` flag.

```bash
cmake -B build -S .
cmake --build build -j
```

After a successful build, `build/infer-seg` is created.

Verify the architecture:

```bash
file build/infer-seg
# ARIES:   ELF 64-bit LSB executable, x86-64, ...
# REGULUS: ELF 64-bit LSB executable, ARM aarch64, ...
```

## Run

A sample image `../rc/cr7.jpg` is bundled with the repo.

### ARIES (same host)

```bash
./build/infer-seg ../../../compilation/instance_segmentation/yolo11m-seg.mxq ../rc/cr7.jpg result.jpg
```

### REGULUS (target board)

Copy `build/infer-seg`, `yolov8m-seg.mxq`, and `../rc/cr7.jpg` to the target board, then:

```bash
chmod +x infer-seg
./infer-seg yolov8m-seg.mxq cr7.jpg result.jpg
```

## Example Output

```
Model input: 640x640x3
Image size: 1920x1080
Inference time: 24.512 ms
Detections: 3
  person 92% [120,45,380,520]
  car 87% [600,200,950,450]
  dog 76% [400,300,550,500]
Result saved to: result.jpg
```

The result image `result.jpg` contains the original image with per-instance
segmentation masks overlaid (alpha-blended in the class color) plus bounding boxes
and class labels.
