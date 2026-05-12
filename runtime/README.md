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

Run MXQ models using the Python `qbruntime` library. ARIES and REGULUS expose the same API; only the environment-preparation steps differ.

### ARIES (x86_64 host with NPU)

Install everything yourself on the host:

#### 1. Driver Installation

Start the Mobilint NPU driver to enable device access on the host. See the [Driver Installation Guide](https://docs.mobilint.com/v1.2/en/installing_driver.html).

If you run inside Docker, expose the NPU to the container:

```bash
--device /dev/aries0:/dev/aries0
```

#### 2. Runtime Library Installation

Install the runtime library in the Python environment ([Runtime Installation Guide](https://docs.mobilint.com/v1.2/en/installing_runtime_library.html)):

```bash
pip install mobilint-qb-runtime
```

#### 3. Additional Dependencies

Depending on the model, you may need extra Python packages (e.g., `torch`, `numpy`, `PIL`, `transformers`). Each model tutorial lists what it needs.

#### 4. Utility Tool (Optional)

Mobilint provides a CLI utility for checking NPU status, verifying MXQ files, and running quick inference. See the [Utility Tool Installation Guide](https://docs.mobilint.com/v1.2/en/installing_utility.html).

### REGULUS (ARM64 target board)

REGULUS boards ship with the driver, the `qbruntime` Python library, and the CLI utility **preinstalled**. Skip steps 1, 2, and 4 above. The only setup the user has to do on-board is step 3 — install any model-specific Python dependencies on the board.

---

## C++ Runtime

The C++ `qbruntime` library runs on both ARIES and REGULUS. The build flow differs by device:

- **ARIES** (x86_64): build the inference binary natively on the host and run it there.
- **REGULUS** (ARM64): cross-compile on an x86_64 host and deploy the resulting binary to the target board.

The [`cpp/`](cpp/README.md) directory covers both flows from the same `CMakeLists.txt` — ARIES native build, and REGULUS cross-compile (toolchain setup, CMake cross-build, on-board deployment).
