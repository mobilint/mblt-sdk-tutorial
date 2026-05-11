# Mobilint Runtime Tutorial

The Mobilint qb runtime is a library for running compiled MXQ models on the Mobilint NPU.
It handles model loading, inference execution, and result retrieval.

<!-- markdownlint-disable MD033 -->
<div align="center">
<img src="../assets/Runtime.avif" width="75%", alt="Runtime Diagram">
</div>
<!-- markdownlint-enable MD033 -->

The runtime is available in two languages: Python and C++.

## Python Runtime

Run MXQ models using the Python `qbruntime` library. Both ARIES and REGULUS share the same API.

### Runtime Preparation

The Mobilint `qbruntime` tutorial assumes you are working on a system or target board equipped with a Mobilint NPU.

> **Note**: The runtime environment does not need to match the compilation environment. It only needs to be a system equipped with a Mobilint NPU (e.g. ARIES) or a target board (e.g. REGULUS). On target boards, the driver, runtime library, and utility tool come preinstalled, so the corresponding installation steps below can be skipped.

#### 1. Driver Installation

After connecting the hardware, start the Mobilint NPU driver to enable device access.
Detailed instructions can be found in the [Driver Installation Guide](https://docs.mobilint.com/v0.29/en/installation.html#driver-installation).

If the driver is successfully installed and you are using Docker, you can enable NPU access inside the container using the following flag:

```bash
--device /dev/aries0:/dev/aries0
```

#### 2. Runtime Library Installation

Next, install the runtime library.
Refer to the [Runtime Installation Guide](https://docs.mobilint.com/v0.29/en/installation.html#runtime-library-installtion) for more information.

To install the runtime library in a Python environment, use the following command:

```bash
pip install mobilint-qb-runtime
```

#### 3. Additional Dependencies

Depending on your model type, you may need additional Python packages (e.g., `torch`, `numpy`, `PIL`, `transformers`). Refer to each specific model tutorial for detailed requirements.

#### 4. Utility Tool (Optional)

Mobilint also provides a utility tool for checking NPU status, verifying MXQ files, and running simple inference tasks.
Refer to the [Utility Tool Installation Guide](https://docs.mobilint.com/v0.29/en/installation.html#utility-installation) for details.

---

## C++ Runtime

The C++ `qbruntime` library runs on both ARIES and REGULUS. The build flow differs by device:

- **ARIES** (x86_64): build the inference binary natively on the host and run it there.
- **REGULUS** (ARM64): cross-compile on an x86_64 host and deploy the resulting binary to the target board.

The [`cpp/`](cpp/README.md) directory currently walks through the REGULUS cross-compilation flow — toolchain setup, CMake cross-build, and on-board deployment.
