# Pose Estimation - C++ Inference (ARIES + REGULUS)

An example of running C++ NPU inference on a single image with skeleton
visualization. The same binary handles both YOLO11m-pose (ARIES) and
YOLOv8m-pose (REGULUS) MXQ models because ultralytics v8/v11 share the same
anchor-free Pose head layout (3 stride x [reg_max*4 box + nc cls +
num_keypoints*3 kpt] channels, with nc=1 and 17 COCO keypoints).

Supports **ARIES native build** (x86_64 host with NPU) and **REGULUS
cross-compile** (x86_64 host -> ARM64 target board) from the same
`CMakeLists.txt`.

## File Structure

- `infer_pose.cc` - Inference binary source (NPU inference, post-processing, box + skeleton visualization)
- `yolo_pose_config.h` - Anchorless YOLO pose configuration (yolo11m-pose / yolov8m-pose P5, single person class, 17 keypoints)
- `utils/` - Shared inference modules (NPURunner, Transformer, YoloPoseDecoder)
- `CMakeLists.txt` - CMake build configuration (host arch auto-detected)

## Prerequisites

- Pick the matching MXQ file from the [compiler tutorial](../../../compilation/pose_estimation/README.md):
  - **ARIES**: `yolo11m-pose.mxq` from `model_compile.py`.
  - **REGULUS**: `yolov8m-pose.mxq` from `model_compile_regulus.py`.

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

After a successful build, `build/infer-pose` is created.

Verify the architecture:

```bash
file build/infer-pose
# ARIES:   ELF 64-bit LSB executable, x86-64, ...
# REGULUS: ELF 64-bit LSB executable, ARM aarch64, ...
```

## Run

A sample image `../rc/cr7.jpg` is bundled with the repo.

### ARIES (same host)

```bash
./build/infer-pose ../../../compilation/pose_estimation/yolo11m-pose.mxq ../rc/cr7.jpg result.jpg
```

### REGULUS (target board)

Copy `build/infer-pose`, `yolov8m-pose.mxq`, and `../rc/cr7.jpg` to the target board, then:

```bash
chmod +x infer-pose
./infer-pose yolov8m-pose.mxq cr7.jpg result.jpg
```

## Example Output

```
Model input: 640x640x3
Image size: 980x652
Inference time: 22.134 ms
Detections: 3
  person 94% [15,88,287,560]
  person 94% [436,61,717,567]
  person 83% [728,151,855,455]
Result saved to: result.jpg
```

The result image `result.jpg` contains the original image with bounding boxes,
class labels, and the 17-keypoint COCO skeleton drawn on each detected person.
