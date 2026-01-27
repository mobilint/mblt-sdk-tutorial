# Mobilint Runtime Tutorial

This section provides detailed instructions for running models using the Mobilint runtime library.

<div align="center">
<img src="../assets/Runtime.avif" width="75%", alt="Runtime Diagram">
</div>

## Runtime Preparation

The Mobilint `qbruntime` tutorial assumes you are working on an inference PC equipped with a Mobilint NPU.

> **Note**: The runtime environment does not need to be the same as the compilation environment. The runtime only requires a system equipped with a Mobilint NPU.

### 1. Driver Installation

After connecting the hardware, start the Mobilint NPU driver to enable device access.
Detailed instructions can be found in the [Driver Installation Guide](https://docs.mobilint.com/v0.29/en/installation.html#driver-installation).

If the driver is successfully installed and you are using Docker, you can enable NPU access inside the container using the following flag:

```bash
--device /dev/aries0:/dev/aries0
```

### 2. Runtime Library Installation

Next, install the runtime library.
Refer to the [Runtime Installation Guide](https://docs.mobilint.com/v0.29/en/installation.html#runtime-library-installtion) for more information.

To install the runtime library in a Python environment, use the following command:

```bash
pip install qbruntime
```

### 3. Additional Dependencies

Depending on your model type, you may need additional Python packages (e.g., `torch`, `numpy`, `PIL`, `transformers`). Refer to each specific model tutorial for detailed requirements.

### 4. Utility Tool (Optional)

Mobilint also provides a utility tool for checking NPU status, verifying MXQ files, and running simple inference tasks.
Refer to the [Utility Tool Installation Guide](https://docs.mobilint.com/v0.29/en/installation.html#utility-installation) for details.

---

## Ready to Run?

You are now ready to run your models!
Explore the tutorials in this directory to run your compiled models on the Mobilint NPU.
