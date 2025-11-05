# Mobilint SDK Tutorial

<div align="center">
<p>
<a href="https://www.mobilint.com/" target="_blank">
<img src="assets/Mobilint_Logo.png" alt="Mobilint Logo" width="60%">
</a>
</p>
</div>

## Overview

This repository provides examples and explanations to help users easily get started with the Mobilint SDK, which includes the compiler (qubee) and the runtime software (maccel) library.

Models converted using the compiler can be executed on the Mobilint NPU through the runtime. When properly configured, this workflow enables models to achieve faster inference performance while maintaining the original model’s accuracy.

## Before you start

Before getting started, ensure that you have access to a Mobilint NPU.
If you don’t have one, please contact [us](mailto:contact@mobilint.com) to discuss evaluation options for your AI application.

The SDK is distributed through the [Mobilint Download Center](https://dl.mobilint.com/). Please sign up for an account before downloading the SDK.

## Getting Started

### Compiler Preparation

The Mobilint qb compiler converts models from popular deep learning frameworks into the Mobilint Model eXeCUtable (MXQ) format.
Using a pre-trained model and a calibration dataset, the compiler parses, quantizes, and optimizes the model for execution on the Mobilint NPU.

<div align="center">
<img src="assets/Compiler.avif" width="75%">
</div>

The qb compiler runs in a Linux environment with Docker installed.
Make sure the following prerequisites are met:

- [Ubuntu](https://ubuntu.com/) 20.04 LTS or later (WSL2 is also supported)
- [Docker](https://docs.docker.com/engine/install/ubuntu/)

If a GPU is available, it is recommended to use it for faster compilation.
In that case, you also need:

- [NVIDIA Driver 535.183.01 or later](https://www.nvidia.com/en-us/drivers/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html#)

After preparing the environment, download the latest qbcompiler image from[qbcompiler Docker Hub](https://hub.docker.com/r/mobilint/qbcompiler). Since version v0.11.\*.\*, two types of images are provided: 

- `{version}-cpu` for CPU compilation
- `{version}-cuda` for GPU compilation

Choose the image that matches your setup.
For example, for version 0.11:

```bash
docker pull mobilint/qbcompiler:0.11-cpu-ubuntu22.04 # CPU compilation
docker pull mobilint/qbcompiler:0.11-cuda12.8.1-ubuntu22.04 # GPU compilation
```

Then, create a Docker container:

```bash
docker run -it --ipc=host -v {path_to_local_workspace}:{path_to_container_workspace} --name {your_container_name} {qbcompiler_docker_image_name}
```

If you want to use GPU for compilation, add the --gpus=all flag.

If your environment also includes a Mobilint NPU and you want to perform both compilation and inference in the same container, add:

```bash
--device /dev/aries0:/dev/aries0
```

to enable the access to the Mobilint NPU. You can connect multiple NPUs by adding `--device /dev/aries1:/dev/aries1`, and so on. Alternatively, you may run multiple containers, each connected to the same NPU with the same device flag.

Example:

```bash
docker run -it --ipc=host \
  -v {path_to_local_workspace}:{path_to_container_workspace} \
  --name {your_container_name} \
  --gpus=all \
  --device /dev/aries0:/dev/aries0 \
  mobilint/qbcompiler:0.11-cuda12.8.1-ubuntu22.04
```

Next, visit the [Mobilint Download Center](https://dl.mobilint.com/) to download the latest qbcompiler wheel file.
After logging in, go to ARIES -> qb Compiler and download the wheel file compatible with your environment.

Copy it to the container and install:

```bash
docker cp {path_to_local_wheel_file} {your_container_name}:{path_to_container_workspace}
docker exec -it {your_container_name} /bin/bash
pip install {path_to_container_workspace}/{wheel_file_name}
```

Verify the installation:

```bash
pip list | grep qubee # Verify the installation
```

Now, you’re ready to compile your models!
Please refer to the [compilation tutorial](compilation/README.md) for detailed instructions.

### Runtime Preparation

The runtime library enables execution of Mobilint-compiled models on the NPU.
Using this library, you can integrate your compiled models into real-world applications.

<div align="center">
<img src="assets/Runtime.avif" width="75%">
</div>

> Note: The environments for the compiler and runtime do not need to be the same. The runtime only requires a system equipped with a Mobilint NPU.

Technically, the enviornment that user run compiler and runtime does not need to be the same. The environment equipped with Mobilint NPU is enough to run the runtime.

After the hardware connection, start the Mobilint NPU driver to enable access to the device.
Detailed instructions can be found in the [Driver installation Guide](https://docs.mobilint.com/v0.29/en/installation.html#driver-installation).

If the driver is successfully installed, you can enable NPU access inside a Docker container using the following flag:

```bash
--device /dev/aries0:/dev/aries0
```

Next, install the runtime library.
Refer to the [Runtime Installation Guide](https://docs.mobilint.com/v0.29/en/installation.html#runtime-library-installtion) for more information.

Mobilint also provides a utility tool for checking NPU status, verifying MXQ files, and running simple inference tasks.
Refer to the [Utility Tool Installation Guide](https://docs.mobilint.com/v0.29/en/installation.html#utility-installation) for details.

Now, you are ready to run your models!
Please refer to the [runtime tutorial](runtime/README.md) for more information.

## Support & Issues

If you encounter any issues while following this tutorial, please contact our [technical support team](mailto:tech-support@mobilint.com).
