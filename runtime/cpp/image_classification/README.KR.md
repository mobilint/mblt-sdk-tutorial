# 이미지 분류 - C++ 추론 (ARIES + REGULUS)

ResNet-50 MXQ 모델로 단일 이미지에 대해 C++ NPU 추론을 실행하는 예제입니다.
같은 `CMakeLists.txt` 로 **ARIES 네이티브 빌드** (x86_64 호스트 + NPU) 와
**REGULUS 크로스 컴파일** (x86_64 호스트 -> ARM64 타겟 보드) 두 경로를 모두 지원합니다.

## 파일 구조

- `infer_cls.cc` - 추론 바이너리 소스 (NPU 추론, Top-5 출력)
- `CMakeLists.txt` - CMake 빌드 설정 (호스트 arch 자동 감지)
- `imagenet_labels.txt` - ImageNet 1000 클래스 라벨 파일

## 사전 준비

- [컴파일러 튜토리얼](../../../compilation/image_classification/README.KR.md) 로 생성한 `resnet50.mxq` 파일이 필요합니다.
  ARIES 는 `model_compile.py`, REGULUS 는 `model_compile_regulus.py` 결과를 사용합니다.

### 공통 요구 사항 (두 경로 모두)

- CMake >= 3.18
- C++17 컴파일러 (gcc / clang)
- `qbruntime` 라이브러리 (Mobilint NPU SDK 설치 시 함께 설치됨)

### ARIES 네이티브 빌드 (x86_64 호스트 + NPU)

호스트에 OpenCV 와 빌드 도구를 설치합니다 (Ubuntu / Debian 기준):

```bash
apt-get update
apt-get install -y build-essential cmake libopencv-dev
```

### REGULUS 크로스 컴파일 (x86_64 호스트 -> ARM64 타겟 보드)

벤더 크로스 컴파일 툴체인에는 OpenCV 와 `qbruntime` 이 포함되어 있습니다. 툴체인을 확인하고 환경을 활성화합니다:

```bash
ls /opt/crosstools/mobilint/                                   # 버전 디렉토리가 보여야 함
unset LD_LIBRARY_PATH                                          # 호스트 CUDA 등 충돌 방지
source /opt/crosstools/mobilint/{version}/{sdk}/environment-setup-cortexa53-mobilint-linux
echo $CXX                                                      # aarch64-mobilint-linux-g++ ...
```

툴체인이 설치되어 있지 않으면 [크로스 컴파일 준비](../README.KR.md) 를 참조해 먼저 설치합니다.

## 빌드

ARIES 네이티브와 REGULUS 크로스 컴파일 모두 같은 명령으로 빌드됩니다.
`CMakeLists.txt` 가 호스트 arch 를 감지해 `-march` 플래그를 자동으로 선택합니다.

```bash
cmake -B build -S .
cmake --build build -j
```

빌드가 끝나면 `build/infer-cls` 바이너리가 생성됩니다.

바이너리의 아키텍처를 확인합니다:

```bash
file build/infer-cls
# ARIES:   ELF 64-bit LSB executable, x86-64, ...
# REGULUS: ELF 64-bit LSB executable, ARM aarch64, ...
```

## 실행

샘플 이미지 `../rc/volcano.jpg` 가 저장소에 함께 들어 있습니다.

### ARIES (같은 호스트)

```bash
./build/infer-cls ../../../compilation/image_classification/resnet50.mxq ../rc/volcano.jpg imagenet_labels.txt
```

### REGULUS (타겟 보드)

`build/infer-cls`, `resnet50.mxq`, `imagenet_labels.txt`, `../rc/volcano.jpg` 를 타겟 보드로 복사한 뒤 실행합니다:

```bash
chmod +x infer-cls
./infer-cls resnet50.mxq volcano.jpg imagenet_labels.txt
```

## 출력 예시

```
Model input: 224x224x3
Inference time: 2.05306 ms

Top-5 predictions:
  980 volcano (30.0102)
  862 torch (11.6544)
  1 goldfish (9.61492)
  469 caldron (8.74083)
  974 geyser (8.74083)
```
