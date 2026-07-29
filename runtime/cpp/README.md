# C++ Runtime

The C++ `qbruntime` library supports both ARIES and REGULUS devices. The inference examples in this directory share the same overall flow, but the build path depends on the target platform:

- **ARIES** (`x86_64`): Build and run the binary directly on the host with NPU access.
- **REGULUS** (`ARM64`): Cross-compile on an `x86_64` host, then deploy the binary to the target board.

This directory focuses on the vision tutorials where inference is a straightforward C++ pipeline: load an MXQ model, preprocess an image, run NPU inference, apply postprocessing, and save or print the result.

## Available Tutorials

- `image_classification/`
- `object_detection/`
- `depth_estimation/`
- `semantic_segmentation/`
- `face_detection/`
- `instance_segmentation/`
- `pose_estimation/`

## Build Overview

Each tutorial directory contains its own `CMakeLists.txt`. The build scripts:

- Require C++17
- Use OpenCV for image loading and visualization
- Link against `qbruntime`
- Switch the optimization flags automatically based on the target architecture

## ARIES Native Build

On ARIES, install the required build tools and OpenCV on the host:

```bash
apt-get update
apt-get install -y build-essential cmake libopencv-dev
```

Then move into the tutorial directory you want to build and run:

```bash
cmake -B build -S .
cmake --build build -j
```

## REGULUS Cross-Compile Setup

On REGULUS, build on an `x86_64` host using the Mobilint cross-compilation toolchain, then copy the resulting binary to the target board.

Download the latest toolchain archive from the [Mobilint Download Center](https://dl.mobilint.com/) under `REGULUS -> Image Archive`, extract it, and run:

```bash
tar -xzf {downloaded_tar_gz_file}
./install-regulus-toolchain.sh
```

After installation, activate the toolchain environment:

```bash
source /opt/crosstools/mobilint/{version}/{sdk}/environment-setup-cortexa53-mobilint-linux
```

If needed, verify that the cross compiler is active:

```bash
echo $CXX
```

You should see an `aarch64-mobilint-linux-g++`-style compiler path.

## REGULUS Build Flow

Inside the tutorial directory, use the same CMake commands:

```bash
cmake -B build -S .
cmake --build build -j
```

Then verify the produced binary:

```bash
file build/<binary-name>
```

For REGULUS, the output should be an `ARM aarch64` executable.

## Notes

- REGULUS target boards usually already include the Mobilint NPU driver and runtime library.
- The per-tutorial READMEs document the expected MXQ file, sample image, binary name, and run command.
