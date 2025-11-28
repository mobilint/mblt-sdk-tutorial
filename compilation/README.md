# Mobilint Compiler Tutorial

Tutorials in this section provide detailed instructions for compiling models using the Mobilint qb compiler.

<div align="center">
<img src="../assets/Compiler.avif" width="75%", alt="Compiler Diagram">
</div>

## Compiler Preparation

The Mobilint qb compiler runs in a Linux environment with Docker installed.
Before starting, ensure you have:

- [Ubuntu](https://ubuntu.com/) 20.04 LTS or later (WSL2 is also supported)
- [Docker](https://docs.docker.com/engine/install/ubuntu/)

If a GPU is available, it is recommended to use it for faster compilation.
In that case, you also need:

- [NVIDIA Driver 535.183.01 or later](https://www.nvidia.com/en-us/drivers/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html#)

After preparing the environment, download the latest qbcompiler image from [qbcompiler Docker Hub](https://hub.docker.com/r/mobilint/qbcompiler). Since version v0.11.\*.\*, two types of images are provided:

- `{version}-cpu*` for compilation on CPU
- `{version}-cuda*` for compilation on GPU with CUDA support

Choose the Docker image that matches your setup.
For example, for version 0.11:

```bash
docker pull mobilint/qbcompiler:0.11-cpu-ubuntu22.04 # compilation on CPU
docker pull mobilint/qbcompiler:0.11-cuda12.8.1-ubuntu22.04 # compilation on GPU with CUDA support
```

Then, create a Docker container:

```bash
docker run -it --ipc=host \
  -v {path_to_local_workspace}:{path_to_container_workspace} \
  --name {your_container_name} \
  {qbcompiler_docker_image_name}
```

If you want to use GPU for compilation, add the `--gpus=all` flag.

If your environment also includes a Mobilint NPU and you want to perform both compilation and inference in the same container, add:

```bash
--device /dev/aries0:/dev/aries0 # Enable the access to the Mobilint NPU on docker container
```

You can connect multiple NPUs by adding `--device /dev/aries1:/dev/aries1`, and so on. Alternatively, you may run multiple containers, each connected to the same NPU with the same device flag.

Example:

```bash
docker run -it --ipc=host \
  -v {path_to_local_workspace}:{path_to_container_workspace} \
  --name {your_container_name} \
  --gpus=all \ # Enable the access to the GPU on docker container
  --device /dev/aries0:/dev/aries0 \ # Enable the access to the Mobilint NPU on docker container
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

Try the tutorials in current directory to compile your models.
