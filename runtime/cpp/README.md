# C++ Runtime

The C++ `qbruntime` library runs on both ARIES and REGULUS. The build flow differs by device:

- **ARIES** (x86_64): build the inference binary natively on the host and run it there.
- **REGULUS** (ARM64): cross-compile the inference binary on an x86_64 host, then deploy it to the target board.

This tutorial walks through both flows from the same `CMakeLists.txt`: the **ARIES native build** on an x86_64 host with the NPU, and the **REGULUS cross-compile flow** (toolchain setup, CMake cross-build, on-board deployment).

> **Note**: The target board ships with the Mobilint NPU driver, runtime library, and utility tool preinstalled, so only the toolchain and cross-build steps below are required.

## Prerequisites (REGULUS target only)

The steps below set up an x86_64 host for cross-compiling binaries that target a REGULUS board. ARIES native builds do **not** require any of this — skip this section if you are only building for ARIES.

### 1. Cross-Compilation Toolchain Installation

Download the latest `tar.gz` file from the [Mobilint Download Center](https://dl.mobilint.com/) under REGULUS -> Image Archive.

Extract and run the toolchain installation script:

```bash
tar -xzf {downloaded_tar_gz_file}
./install-regulus-toolchain.sh
```

Once installed, activate the cross-compilation environment:

```bash
source /opt/crosstools/mobilint/{version}/environment-setup-cortexa53-mobilint-linux
```
