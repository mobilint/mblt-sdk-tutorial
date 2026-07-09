# Python Runtime

The Python `qbruntime` library exposes the same Mobilint NPU API on both ARIES and REGULUS. This page covers the Python-specific setup for the tutorials in this directory. For general runtime installation and device setup, see the [runtime overview](../README.md).

## Quick Start

Follow these steps before running any Python tutorial in this directory.

### 1. Enable the NPU Driver

Make sure the Mobilint NPU driver is installed and running on the host. If it is not installed yet, follow the [Driver Installation Guide](https://docs.mobilint.com/v1.2/en/installing_driver.html).

If you are running inside Docker, expose the device to the container:

```bash
--device /dev/aries0:/dev/aries0
```

### 2. Install the Python Runtime Library

```bash
pip install mobilint-qb-runtime
```

### 3. Install Tutorial-Specific Dependencies

Each tutorial directory documents its own Python dependencies. Depending on the model, you may need packages such as `numpy`, `Pillow`, `torch`, `transformers`, or `mblt-model-zoo`.

Install the dependencies listed in the README for the tutorial you want to run, for example:

- `image_classification/`
- `object_detection/`
- `bert/`
- `llm/`
- `stt/`
- `vlm/`

### 4. Run the Tutorial Script

Move into the tutorial directory you want to use and run the documented script from there.

## REGULUS Preinstalled Environment

REGULUS target boards usually ship with the driver, `qbruntime`, and the utility tools already installed. In that environment, you can usually skip steps 1 and 2 and start from the tutorial-specific dependencies.

## Device Recommendation

- **ARIES** (`x86_64`): Recommended. Host-side preprocessing and postprocessing are less likely to become the bottleneck.
- **REGULUS** (`ARM64`): Supported, but Python workloads can be slow because preprocessing, postprocessing, and tensor manipulation may dominate end-to-end latency.

For production-style workloads on REGULUS, prefer the [C++ runtime](../cpp/README.md) when possible.
