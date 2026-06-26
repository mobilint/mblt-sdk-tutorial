# Face Detection - C++ Inference (ARIES + REGULUS)

An example of running C++ NPU inference on a single image with bounding-box
visualization for faces. The model is a single-class (nc=1, "face") YOLO
anchor-free Detect head (e.g. YOLOv12m-face) whose output layout matches the
generic ultralytics Detect head (3 stride x [reg_max*4 box + nc cls] = 195
channels), so the same DFL decode + NMS pipeline used for object detection
applies. Outputs are plain bounding boxes; there are no landmark/keypoint
channels.

Supports **ARIES native build** (x86_64 host with NPU) and **REGULUS
cross-compile** (x86_64 host -> ARM64 target board) from the same
`CMakeLists.txt`.

## File Structure

- `infer_face.cc` - Inference binary source (NPU inference, post-processing, bbox visualization)
- `yolo_face_config.h` - Anchorless YOLO face configuration (yolov12m-face P5, 1 class)
- `utils/` - Shared inference modules (NPURunner, Transformer, YoloDecoder)
- `CMakeLists.txt` - CMake build configuration (host arch auto-detected)

## Prerequisites

- Pick the matching MXQ file from the [compiler tutorial](../../../compilation/face_detection/README.md):
  - `yolov12m-face.mxq`.

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

After a successful build, `build/infer-face` is created.

Verify the architecture:

```bash
file build/infer-face
# ARIES:   ELF 64-bit LSB executable, x86-64, ...
# REGULUS: ELF 64-bit LSB executable, ARM aarch64, ...
```

## Run

A sample image `../rc/cr7.jpg` is bundled with the repo.

### ARIES (same host)

```bash
./build/infer-face ../../../compilation/face_detection/yolov12m-face.mxq ../rc/cr7.jpg result.jpg
```

### REGULUS (target board)

Copy `build/infer-face`, `yolov12m-face.mxq`, and `../rc/cr7.jpg` to the target board, then:

```bash
chmod +x infer-face
./infer-face yolov12m-face.mxq cr7.jpg result.jpg
```

## Example Output

```
Model input: 640x640x3
Image size: 980x652
Inference time: 18.342 ms
Detections: 2
  face 98% [430,58,712,420]
  face 94% [18,90,280,470]
Result saved to: result.jpg
```

The result image `result.jpg` contains the original image with face bounding boxes and labels drawn on it.
