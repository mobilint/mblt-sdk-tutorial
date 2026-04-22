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

Run MXQ models using the Python `qbruntime` library. Both Aries and Regulus can perform inference with the same API.

- `image_classification/` - Image Classification (ResNet-50)
- `object_detection/` - Object Detection (YOLO)
- `instance_segmentation/` - Instance Segmentation
- `pose_estimation/` - Pose Estimation
- `face_detection/` - Face Detection
- `bert/` - BERT Embedding
- `llm/` - Large Language Model
- `stt/` - Speech-to-Text
- `tts/` - Text-to-Speech
- `vlm/` - Vision Language Model

### Runtime Preparation

The Mobilint `qbruntime` tutorial assumes you are working on a system equipped with a Mobilint NPU.

> **Note**: The runtime environment does not need to be the same as the compilation environment. The runtime only requires a system equipped with a Mobilint NPU.

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

## Cross-Compilation (C++)

For environments where the host (x86_64) and target board (ARM64) are separate, such as Regulus, you can cross-compile C++ inference binaries optimized for the target.
The built binaries are then transferred to the target board for execution.
This tutorial covers the process of building inference binaries using the C++ `qbruntime`.
For toolchain installation, CMake cross-build, and target board deployment, refer to the [`cross_compile/`](cross_compile/README.md) directory.
