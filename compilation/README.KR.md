# Mobilint 컴파일러 튜토리얼

이 섹션의 튜토리얼은 Mobilint qb 컴파일러를 사용하여 모델을 컴파일하는 방법에 대한 안내를 제공합니다.

<div align="center">
<img src="../assets/Compiler.avif" width="75%", alt="Compiler Diagram">
</div>

## 컴파일러 준비

Mobilint qb 컴파일러는 Docker가 설치된 Linux 환경에서 실행됩니다.
시작하기 전에 다음이 준비되어 있는지 확인하세요:

- [Ubuntu](https://ubuntu.com/) 20.04 LTS 이상 (WSL2도 지원됨)
- [Docker](https://docs.docker.com/engine/install/ubuntu/)

GPU를 사용할 수 있는 경우, 더 빠른 컴파일을 위해 GPU 사용을 권장합니다.
이 경우 다음의 패키지들도 필요합니다:

- [NVIDIA Driver 535.183.01 이상](https://www.nvidia.com/en-us/drivers/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html#)

환경설정을 완료한 후, [qbcompiler Docker Hub](https://hub.docker.com/r/mobilint/qbcompiler)에서 최신 qbcompiler 이미지를 다운로드해야 합니다. qb compiler 버전 v0.11.\*.\*부터 해당 버전과 호환되는 두 가지 유형의 이미지가 제공됩니다:

- `{version}-cpu*` - CPU에서 컴파일 시 사용
- `{version}-cuda*` - CUDA 지원 GPU에서 컴파일 시 사용

설정에 맞는 Docker 이미지를 선택해야 합니다.
예를 들어, 버전 0.11의 경우:

```bash
docker pull mobilint/qbcompiler:0.11-cpu-ubuntu22.04 # CPU에서 컴파일
docker pull mobilint/qbcompiler:0.11-cuda12.8.1-ubuntu22.04 # CUDA 지원 GPU에서 컴파일
```

그런 다음 Docker 컨테이너를 생성해야 합니다:

```bash
docker run -it --ipc=host \
  -v {path_to_local_workspace}:{path_to_container_workspace} \
  --name {your_container_name} \
  {qbcompiler_docker_image_name}
```

컴파일 시 GPU를 사용하려면 `--gpus=all` 플래그를 추가해야 합니다.

컴파일 환경에 Mobilint NPU가 장착되어 있고 동일한 컨테이너에서 컴파일과 추론을 모두 수행하려면 다음의 플래그를 추가해야 합니다:

```bash
--device /dev/aries0:/dev/aries0 # Docker 컨테이너에서 Mobilint NPU 접근 활성화
```

`--device /dev/aries1:/dev/aries1` 등을 추가하여 여러 NPU를 연결할 수 있습니다. 또는 동일한 디바이스 플래그로 동일한 NPU에 연결된 여러 컨테이너를 실행할 수 있습니다.

예시:

```bash
docker run -it --ipc=host \
  -v {path_to_local_workspace}:{path_to_container_workspace} \
  --name {your_container_name} \
  --gpus=all \ # Docker 컨테이너에서 GPU 접근 활성화
  --device /dev/aries0:/dev/aries0 \ # Docker 컨테이너에서 Mobilint NPU 접근 활성화
  mobilint/qbcompiler:0.11-cuda12.8.1-ubuntu22.04
```

다음으로, [Mobilint 다운로드 센터](https://dl.mobilint.com/)를 방문하여 최신 qbcompiler wheel 파일을 다운로드해야 합니다.
로그인 후, ARIES -> qb Compiler로 이동하여 환경에 호환되는 wheel 파일을 다운로드하세요.

컨테이너로 복사하고 설치해야 합니다:

```bash
docker cp {path_to_local_wheel_file} {your_container_name}:{path_to_container_workspace}
docker exec -it {your_container_name} /bin/bash
pip install {path_to_container_workspace}/{wheel_file_name}
```

설치를 확인해야 합니다:

```bash
pip list | grep qubee # 설치 확인
```

이제 모델을 컴파일할 준비가 되었습니다!

현재 디렉토리의 튜토리얼을 따라 모델을 컴파일해 보세요.
