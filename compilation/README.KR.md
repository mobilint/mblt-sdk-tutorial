# Mobilint 컴파일러 튜토리얼 (Mobilint Compiler Tutorial)

이 섹션에는 Mobilint `qbcompiler`로 모델을 컴파일하는 방법을 단계별로 설명한 튜토리얼이 정리되어 있습니다.

<!-- markdownlint-disable MD033 -->
<div align="center">
<img src="../assets/Compiler.avif" width="75%" alt="Compiler Diagram">
</div>
<!-- markdownlint-enable MD033 -->

## 컴파일러 준비 (Compiler Preparation)

Mobilint `qbcompiler`는 Docker가 설치된 Linux 환경에서 실행됩니다.
시작하기 전에 다음 항목이 준비되어 있는지 확인하세요.

- [Ubuntu](https://ubuntu.com/) 20.04 LTS 이상 (WSL2 지원)
- [Docker](https://docs.docker.com/engine/install/ubuntu/)

GPU를 사용할 수 있다면 컴파일 시간을 줄이기 위해 사용하는 것이 좋습니다.
이 경우 다음 항목도 추가로 필요합니다.

- [NVIDIA Driver 535.183.01 이상](https://www.nvidia.com/en-us/drivers/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html#)

환경 준비가 완료되면 [qbcompiler Docker Hub](https://hub.docker.com/r/mobilint/qbcompiler)에서 최신 qbcompiler 이미지를 다운로드하세요. 버전 0.11.x부터 두 가지 유형의 이미지가 제공됩니다:

- `{version}-cpu*`: CPU에서 컴파일 시 사용
- `{version}-cuda*`: CUDA를 지원하는 GPU에서 컴파일 시 사용

사용 중인 환경에 맞는 Docker 이미지를 선택하세요.
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

### ARIES

환경에 Mobilint NPU가 포함되어 있고 동일한 컨테이너에서 컴파일과 추론을 모두 수행하려는 경우, 다음 플래그를 추가하세요:

```bash
--device /dev/aries0:/dev/aries0 # Docker 컨테이너에서 Mobilint NPU 접근 활성화
```

`--device /dev/aries1:/dev/aries1` 등을 추가하여 여러 NPU를 연결할 수 있습니다. 또는 동일한 디바이스 플래그를 사용하여 동일한 NPU에 연결된 여러 컨테이너를 실행할 수도 있습니다.

예시:

> `--gpus=all`은 선택 사항입니다. GPU가 없는 환경에서는 생략해도 CPU 컴파일이 가능합니다.

```bash
docker run -it --ipc=host \
  -v {path_to_local_workspace}:{path_to_container_workspace} \
  --name {your_container_name} \
  --gpus=all \
  --device /dev/aries0:/dev/aries0 \
  mobilint/qbcompiler:1.0-cuda12.8.1-ubuntu22.04
```

다음으로, [Mobilint 다운로드 센터](https://dl.mobilint.com/)를 방문하여 최신 qbcompiler wheel 파일을 다운로드하세요.
로그인한 뒤 `ARIES -> qb Compiler` 메뉴에서 현재 환경에 맞는 wheel 파일을 다운로드하세요.

### REGULUS

REGULUS는 호스트(x86_64)에서 컴파일하고 타겟 보드에서 추론하는 크로스 컴파일 방식입니다.
컴파일 시 NPU 디바이스를 Docker 컨테이너에 연결할 필요가 없습니다.

예시:

> `--gpus=all`은 선택 사항입니다. GPU가 없는 환경에서는 생략해도 CPU 컴파일이 가능합니다.

```bash
docker run -it --ipc=host \
  -v {path_to_local_workspace}:{path_to_container_workspace} \
  --name {your_container_name} \
  --gpus=all \
  mobilint/qbcompiler:1.0-cuda12.8.1-ubuntu22.04
```

다음으로, [Mobilint 다운로드 센터](https://dl.mobilint.com/)를 방문하여 최신 qbcompiler wheel 파일을 다운로드하세요.
로그인한 뒤 `REGULUS -> qb Compiler` 메뉴에서 현재 환경에 맞는 wheel 파일을 다운로드하세요.

다운로드한 파일을 컨테이너로 복사하고 설치합니다:

```bash
docker cp {path_to_local_wheel_file} {your_container_name}:{path_to_container_workspace}
docker exec -it {your_container_name} /bin/bash
pip install {path_to_container_workspace}/{wheel_file_name}
```

REGULUS 컴파일을 시작하기 전에, 호스트 라이브러리 override를 초기화하고 Mobilint 크로스 컴파일 환경을 활성화하세요:

```bash
unset LD_LIBRARY_PATH
source /opt/crosstools/mobilint/{version}/{sdk}/environment-setup-cortexa53-mobilint-linux
```

이 `unset LD_LIBRARY_PATH` 단계는 CUDA, conda, 기타 호스트 라이브러리가 이미 환경 변수에 들어 있는 `x86_64` 호스트에서 특히 중요합니다. 이 값을 그대로 두면 해당 라이브러리가 REGULUS 크로스 툴체인으로 섞여 들어갈 수 있습니다.

설치된 내용을 확인합니다:

```bash
pip list | grep qbcompiler # 설치 확인
```

이제 모델을 컴파일할 준비가 되었습니다.

이 디렉토리의 튜토리얼을 이어서 따라가며 모델별 컴파일 절차를 진행하세요.
