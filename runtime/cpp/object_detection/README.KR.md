# 객체 탐지 - C++ 크로스 컴파일 추론

YOLOv9m MXQ 모델을 사용하여 단일 이미지에 대해 C++ NPU 추론을 실행하고, 바운딩 박스를 시각화하는 예제.

## 파일 구조

- `infer_det.cc` - 추론 바이너리 소스 (NPU 추론, 후처리, bbox 시각화)
- `yolov9m_config.h` - YOLOv9m 모델 설정 (전처리, 후처리 파라미터)
- `utils/` - 공유 추론 모듈 (NPURunner, Transformer, YoloDecoder)
- `CMakeLists.txt` - CMake 빌드 설정

## 사전 준비

- [컴파일러 튜토리얼](../../../compilation/object_detection/README.KR.md)의 REGULUS 섹션에서 `model_compile_regulus.py`로 생성한 `yolov9m.mxq` 파일이 필요하다.

### 1. 툴체인 설치 확인

크로스 컴파일 툴체인이 설치되어 있는지 확인한다:

```bash
ls /opt/crosstools/mobilint/
```

설치된 버전 디렉토리(예: `1.0.0/`)가 보여야 한다. 설치되어 있지 않다면 [크로스 컴파일 준비](../README.KR.md)를 참조하여 툴체인을 먼저 설치한다.

### 2. 크로스 컴파일 환경 활성화

설치된 툴체인 디렉토리 안에서 `environment-setup-cortexa53-mobilint-linux` 파일을 찾는다:

```bash
find /opt/crosstools/mobilint/ -name "environment-setup-cortexa53-mobilint-linux"
# /opt/crosstools/mobilint/1.0.0/v3.4.0/environment-setup-cortexa53-mobilint-linux
```

호스트의 라이브러리 경로(CUDA 등)가 크로스 컴파일러와 충돌할 수 있으므로 먼저 해제한다:

```bash
unset LD_LIBRARY_PATH
```

위에서 찾은 경로로 툴체인 환경을 활성화한다:

```bash
source /opt/crosstools/mobilint/1.0.0/v3.4.0/environment-setup-cortexa53-mobilint-linux
```

활성화 후 크로스 컴파일러가 설정되었는지 확인한다:

```bash
echo $CXX
# aarch64-mobilint-linux-g++ ...
```

## 빌드 (호스트)

```bash
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE
make -j$(nproc)
cd ..
```

빌드가 완료되면 `build/infer-det` 바이너리가 생성된다.

바이너리가 ARM64용인지 확인한다:

```bash
file build/infer-det
# build/infer-det: ELF 64-bit LSB executable, ARM aarch64, ...
```

## 실행 (타겟 보드)

`build/infer-det` 바이너리와 `yolov9m.mxq` 모델 파일을 타겟 보드에 복사한 뒤 실행한다:

```bash
chmod +x infer-det
./infer-det yolov9m.mxq example.jpg result.jpg
```

## 출력 예시

```
Model input: 640x640x3
Image size: 1920x1080
Inference time: 15.234 ms
Detections: 3
  person 92% [120,45,380,520]
  car 87% [600,200,950,450]
  dog 76% [400,300,550,500]
Result saved to: result.jpg
```

`result.jpg`에 바운딩 박스와 클래스 이름이 그려진 결과 이미지가 저장된다.
