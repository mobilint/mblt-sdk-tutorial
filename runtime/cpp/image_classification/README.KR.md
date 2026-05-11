# 이미지 분류 - C++ 크로스 컴파일 추론

ResNet-50 MXQ 모델을 사용하여 단일 이미지에 대해 C++ NPU 추론을 실행하는 예제입니다.

## 파일 구조

- `infer_cls.cc` - 추론 바이너리 소스 (NPU 추론, Top-5 출력)
- `CMakeLists.txt` - CMake 빌드 설정
- `imagenet_labels.txt` - ImageNet 1000 클래스 라벨 파일

## 사전 준비

- [컴파일러 튜토리얼](../../../compilation/image_classification/README.KR.md)의 REGULUS 섹션에서 `model_compile_regulus.py`로 생성한 `resnet50.mxq` 파일이 필요합니다.

### 1. 툴체인 설치 확인

크로스 컴파일 툴체인이 설치되어 있는지 확인합니다:

```bash
ls /opt/crosstools/mobilint/
```

설치된 버전 디렉토리(예: `1.0.0/`)가 보여야 합니다. 설치되어 있지 않다면 [크로스 컴파일 준비](../README.KR.md)를 참조하여 툴체인을 먼저 설치하세요.

### 2. 크로스 컴파일 환경 활성화

설치된 툴체인 디렉토리 안에서 `environment-setup-cortexa53-mobilint-linux` 파일을 찾습니다:

```bash
find /opt/crosstools/mobilint/ -name "environment-setup-cortexa53-mobilint-linux"
# /opt/crosstools/mobilint/1.0.0/v3.4.0/environment-setup-cortexa53-mobilint-linux
```

호스트의 라이브러리 경로(CUDA 등)가 크로스 컴파일러와 충돌할 수 있으므로 먼저 해제합니다:

```bash
unset LD_LIBRARY_PATH
```

위에서 찾은 경로로 툴체인 환경을 활성화합니다:

```bash
source /opt/crosstools/mobilint/1.0.0/v3.4.0/environment-setup-cortexa53-mobilint-linux
```

활성화 후 크로스 컴파일러가 설정되었는지 확인합니다:

```bash
echo $CXX
# aarch64-mobilint-linux-g++ ...
```

## 빌드 (호스트)

```bash
mkdir build && cd build
cmake ..
make -j8
cd ..
```

빌드가 완료되면 `build/infer-cls` 바이너리가 생성됩니다.

바이너리가 ARM64용인지 확인합니다:

```bash
file build/infer-cls
# build/infer-cls: ELF 64-bit LSB executable, ARM aarch64, ...
```

## 실행 (타겟 보드)

`build/infer-cls` 바이너리, `resnet50.mxq` 모델 파일, `imagenet_labels.txt` 라벨 파일을 타겟 보드에 복사한 뒤 실행합니다:

```bash
chmod +x infer-cls
./infer-cls resnet50.mxq example.jpg imagenet_labels.txt
```

## 출력 예시

```
Model input: 224x224x3
Inference time: 4.42336 ms

Top-5 predictions:
  980 volcano (12.345)
  970 alp (10.234)
  928 ice cream (8.567)
  949 strawberry (7.890)
  654 miniskirt (6.123)
```
