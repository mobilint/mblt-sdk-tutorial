# C++ 런타임

C++ `qbruntime` 라이브러리는 ARIES와 REGULUS를 모두 지원합니다. 이 디렉토리의 추론 예제는 전반적인 흐름은 같지만, 빌드 방식은 타겟 플랫폼에 따라 달라집니다.

- **ARIES** (`x86_64`): NPU에 접근 가능한 호스트에서 바이너리를 직접 빌드하고 실행합니다.
- **REGULUS** (`ARM64`): `x86_64` 호스트에서 크로스 컴파일한 뒤 바이너리를 타겟 보드로 배포해 실행합니다.

이 디렉토리는 비전 튜토리얼 중심으로 구성되어 있습니다. 공통 흐름은 C++에서 MXQ 모델을 로드하고, 이미지를 전처리한 뒤, NPU 추론을 실행하고, 후처리를 적용해 결과를 저장하거나 출력하는 방식입니다.

## 제공 튜토리얼

- `image_classification/`
- `object_detection/`
- `depth_estimation/`
- `semantic_segmentation/`
- `face_detection/`
- `instance_segmentation/`
- `pose_estimation/`

## 빌드 개요

각 튜토리얼 디렉토리에는 자체 `CMakeLists.txt`가 포함되어 있습니다. 빌드 스크립트는 다음을 공통으로 사용합니다.

- C++17
- 이미지 로딩과 시각화를 위한 OpenCV
- `qbruntime` 링크
- 타겟 아키텍처에 맞는 최적화 플래그 자동 선택

## ARIES 네이티브 빌드

ARIES에서는 호스트에 필요한 빌드 도구와 OpenCV를 설치한 뒤 빌드합니다.

```bash
apt-get update
apt-get install -y build-essential cmake libopencv-dev
```

그 다음 원하는 튜토리얼 디렉토리로 이동해 다음 명령으로 빌드하고 실행할 수 있습니다.

```bash
cmake -B build -S .
cmake --build build -j
```

## REGULUS 크로스 컴파일 준비

REGULUS에서는 `x86_64` 호스트에서 Mobilint 크로스 컴파일 툴체인을 사용해 빌드한 뒤, 생성된 바이너리를 타겟 보드로 복사해 실행합니다.

[Mobilint 다운로드 센터](https://dl.mobilint.com/)의 `REGULUS -> Image Archive`에서 최신 툴체인 아카이브를 내려받아 압축을 풀고 다음 명령을 실행하세요.

```bash
tar -xzf {downloaded_tar_gz_file}
./install-regulus-toolchain.sh
```

설치가 끝나면 툴체인 환경을 활성화합니다.

```bash
source /opt/crosstools/mobilint/{version}/{sdk}/environment-setup-cortexa53-mobilint-linux
```

필요하다면 다음 명령으로 크로스 컴파일러가 활성화되었는지 확인할 수 있습니다.

```bash
echo $CXX
```

정상이라면 `aarch64-mobilint-linux-g++` 형태의 컴파일러 경로가 출력됩니다.

## REGULUS 빌드 흐름

튜토리얼 디렉토리 안에서 ARIES와 동일하게 CMake 명령을 실행합니다.

```bash
cmake -B build -S .
cmake --build build -j
```

생성된 바이너리는 다음 명령으로 확인할 수 있습니다.

```bash
file build/<binary-name>
```

REGULUS용이라면 결과가 `ARM aarch64` 실행 파일이어야 합니다.

## 참고

- REGULUS 타겟 보드에는 보통 Mobilint NPU 드라이버와 런타임 라이브러리가 이미 설치되어 있습니다.
- 각 튜토리얼 README에는 필요한 MXQ 파일, 샘플 이미지, 바이너리 이름, 실행 명령이 정리되어 있습니다.
