# Python Runtime

The Python `qbruntime` library exposes the same NPU API on both ARIES and REGULUS. See the [runtime overview](../README.md) for the general driver and runtime-library setup; this document covers only the Python-specific steps.

## Usage Steps

#### 1. Enable the Driver

Make sure the Mobilint NPU driver is running on the host. If it is not installed, follow the [Driver Installation Guide](https://docs.mobilint.com/v1.2/en/installing_driver.html). When running inside Docker, expose the NPU to the container with `--device /dev/aries0:/dev/aries0`.

#### 2. Install the Python Runtime Library

```bash
pip install mobilint-qb-runtime
```

#### 3. Install Model-specific Dependencies

Install the packages required by each model tutorial (`image_classification/`, `object_detection/`, `llm/`, `stt/`, ...) — typically some subset of `numpy`, `PIL`, `torch`, `transformers`. The exact list is documented in each subdirectory README.

#### 4. Run the Script

Move into the desired model directory and run its inference script.

## REGULUS Preinstalled

REGULUS target boards ship with the driver, the `qbruntime` library, and the utility tool already installed. Skip steps 1 and 2 and start from step 3 (model-specific dependencies).

## Device Recommendation

- **ARIES** (x86_64): **recommended**. The x86_64 host has enough CPU headroom and a complete Python ecosystem, so NPU inference is rarely bottlenecked by host-side preprocessing or postprocessing.
- **REGULUS** (ARM64): **supported, but can be very slow**. The Cortex-A53 host CPU is far weaker than a typical x86_64 host, so Python-level preprocessing, postprocessing, and tensor manipulation (`numpy`, `torch`) often dominate end-to-end latency even when NPU inference itself is fast. For production workloads on REGULUS, use the [C++ runtime](../cpp/README.md) instead.
