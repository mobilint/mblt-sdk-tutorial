# Cross-Compilation Runtime (C++)

This section covers cross-compiling C++ inference binaries on the host (x86_64) and running them on the target board (ARM64).

## Prerequisites

### 1. Cross-Compilation Toolchain Installation

Download the latest `tar.gz` file from the [Mobilint Download Center](https://dl.mobilint.com/) under REGULUS -> Image Archive.

Extract and run the toolchain installation script:

```bash
tar -xzf {downloaded_tar_gz_file}
sudo ./install-regulus-toolchain.sh
```

Once installed, activate the cross-compilation environment:

```bash
source /opt/crosstools/mobilint/{version}/environment-setup-cortexa53-mobilint-linux
```
