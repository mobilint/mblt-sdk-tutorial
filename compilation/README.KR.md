# Mobilint 컴파일러 튜토리얼 (Mobilint Compiler Tutorial)

이 섹션의 튜토리얼은 Mobilint qb 컴파일러를 사용하여 모델을 컴파일하는 방법에 대한 상세한 안내를 제공합니다.

<div align="center">
<img src="../assets/Compiler.avif" width="75%" alt="Compiler Diagram">
</div>

## 컴파일러 준비 (Compiler Preparation)

Mobilint qb 컴파일러는 Docker가 설치된 Linux 환경에서 실행됩니다.
시작하기 전에 다음 사항이 준비되어 있는지 확인하세요:

- [Ubuntu](https://ubuntu.com/) 20.04 LTS 이상 (WSL2 지원)
- [Docker](https://docs.docker.com/engine/install/ubuntu/)

GPU를 사용할 수 있는 경우, 더 빠른 컴파일을 위해 GPU 사용을 권장합니다.
이 경우 다음 항목들이 추가로 필요합니다:

- [NVIDIA Driver 535.183.01 이상](https://www.nvidia.com/en-us/drivers/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html#)

환경 준비가 완료되면 [qbcompiler Docker Hub](https://hub.docker.com/r/mobilint/qbcompiler)에서 최신 qbcompiler 이미지를 다운로드하세요. 버전 0.11.x부터 두 가지 유형의 이미지가 제공됩니다:

- `{version}-cpu*`: CPU에서 컴파일 시 사용
- `{version}-cuda*`: CUDA를 지원하는 GPU에서 컴파일 시 사용

사용자 환경에 맞는 Docker 이미지를 선택하세요.
예를 들어, 버전 1.0.1의 경우 다음과 같습니다:

```bash
docker pull mobilint/qbcompiler:1.0-cpu-ubuntu22.04 # CPU에서 컴파일
docker pull mobilint/qbcompiler:1.0-cuda12.8.1-ubuntu22.04 # CUDA 지원 GPU에서 컴파일
```

그 다음, Docker 컨테이너를 생성합니다:

```bash
docker run -it --ipc=host \
  -v {path_to_local_workspace}:{path_to_container_workspace} \
  --name {your_container_name} \
  {qbcompiler_docker_image_name}
```

컴파일 시 GPU를 사용하려면 `--gpus=all` 플래그를 추가하세요.

환경에 Mobilint NPU가 포함되어 있고 동일한 컨테이너에서 컴파일과 추론을 모두 수행하려는 경우, 다음 플래그를 추가하세요:

```bash
--device /dev/aries0:/dev/aries0 # Docker 컨테이너에서 Mobilint NPU 접근 활성화
```

`--device /dev/aries1:/dev/aries1` 등을 추가하여 여러 NPU를 연결할 수 있습니다. 또는 동일한 디바이스 플래그를 사용하여 동일한 NPU에 연결된 여러 컨테이너를 실행할 수도 있습니다.

예시:

```bash
docker run -it --ipc=host \
  -v {path_to_local_workspace}:{path_to_container_workspace} \
  --name {your_container_name} \
  --gpus=all \ # Docker 컨테이너에서 GPU 접근 활성화
  --device /dev/aries0:/dev/aries0 \ # Docker 컨테이너에서 Mobilint NPU 접근 활성화
  mobilint/qbcompiler:1.0-cuda12.8.1-ubuntu22.04
```

다음으로, [Mobilint 다운로드 센터](https://dl.mobilint.com/)를 방문하여 최신 qbcompiler wheel 파일을 다운로드하세요.
로그인 후, ARIES -> qb Compiler 메뉴에서 사용자 환경과 호환되는 wheel 파일을 다운로드할 수 있습니다.

다운로드한 파일을 컨테이너로 복사하고 설치합니다:

```bash
docker cp {path_to_local_wheel_file} {your_container_name}:{path_to_container_workspace}
docker exec -it {your_container_name} /bin/bash
pip install {path_to_container_workspace}/{wheel_file_name}
```

설치된 내용을 확인합니다:

```bash
pip list | grep qbcompiler # 설치 확인
```

이제 모델을 컴파일할 준비가 되었습니다!

현재 디렉토리의 튜토리얼을 따라 모델을 컴파일해 보세요.
