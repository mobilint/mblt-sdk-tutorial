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

Models converted using the compiler can be executed on the Mobilint NPU through the runtime. When properly configured, this process enables models to achieve faster inference performance while maintaining the accuracy of the original model.

## Before you start

Before getting started, you need to prepare Mobilint NPU. If you don't have one, please contact [us](mailto:contact@mobilint.com) to get assessment for your AI application.

We distribute our SDK through [Mobilint Download Center](https://dl.mobilint.com/). Please sign up for an account before downloading the SDK.

## Getting Started

### Compiler Preparation

Mobilint's qb compiler works on Linux environment with Docker installed. So, prepare the following prerequisites:

- [Ubuntu](https://ubuntu.com/) 20.04 LTS or later (WSL2 is also supported)
- [Docker](https://docs.docker.com/engine/install/ubuntu/)

In adition, if GPU is available, it is recommended to use GPU for compilation to speed up the process with the following requirements:
- [NVIDIA Driver 535.183.01 or later](https://www.nvidia.com/en-us/drivers/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html#)

After preparing the prerequisites, visit [qbcompiler Docker Hub](https://hub.docker.com/r/mobilint/qbcompiler) to download the latest qbcompiler image. Since the release of qbcompiler v0.11.\*.\*, we distribute the qbcompiler images that start with `{version}-cpu` and `{version}-cuda` suffixes, for CPU and GPU compilation respectively. So, download the image with the appropriate suffix based on your environment.

For example, for version 0.11, you may use the following commands.

```bash
docker pull mobilint/qbcompiler:0.11-cpu-ubuntu22.04 # CPU compilation
docker pull mobilint/qbcompiler:0.11-cuda12.8.1-ubuntu22.04 # GPU compilation
```

Then, build the Docker container with the following command.

```bash
docker run -it --ipc=host -v {path_to_local_workspace}:{path_to_container_workspace} --name {your_container_name} {qbcompiler_docker_image_name}
```

If you want to use GPU for compilation, you need to add `--gpus=all` flag to enable GPU access.

In addition, if your environment is equipped with Mobilint NPU and you want to proceed the compilation and inference with the same container, you need to add `--device /dev/aries0:/dev/aries0` flag to enable the access to the Mobilint NPU. Besides, you may connect multiple NPUs by adding `--device /dev/aries1:/dev/aries1` and so on.

For example, for version 0.11, you may use the following commands.

```bash
docker run -it --ipc=host -v {path_to_local_workspace}:{path_to_container_workspace} --name {your_container_name} --gpus=all --device /dev/aries0:/dev/aries0 mobilint/qbcompiler:0.11-cuda12.8.1-ubuntu22.04
```

Then, visit our Download Center to download the latest qbcompiler wheel file. After logging in, go to ARIES Tab -> qb Compiler and download the wheel file that is compatible with your environment.

### Runtime Preparation

Runtime tutorial should work with environment equipped with Mobilint NPU. 

## Compilation

<center><img src="assets/Compiler.avif" width="75%"></center>

## Runtime

<center><img src="assets/Runtime.avif" width="75%"></center>

Runtime tutorial includes 

## Support & Issues

If you encounter any problems with this tutorial, please feel free to contact [us](mailto:tech-support@mobilint.com).
