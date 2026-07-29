# C++ 깊이 추정 런타임

이 튜토리얼은 C++ `qbruntime` API로 컴파일된 `YOLO26m-depth` MXQ 모델을 실행하고 컬러 depth overlay 이미지를 저장하는 방법을 설명합니다.

시작하기 전에 [../../../compilation/depth_estimation/README.KR.md](../../../compilation/depth_estimation/README.KR.md)의 컴파일 과정을 완료하세요. 아래 예제는 `../../../compilation/depth_estimation/yolo26m-depth.mxq`를 사용합니다.

## 사전 준비

다음 구성 요소가 준비되어 있어야 합니다.

- Mobilint NPU 드라이버 및 C++ `qbruntime`
- OpenCV 개발 라이브러리
- C++17 컴파일러
- CMake `3.21` 이상
- 컴파일된 `yolo26m-depth.mxq` 모델

Ubuntu 또는 Debian 기반 ARIES 네이티브 빌드에서는 다음과 같이 설치할 수 있습니다.

```bash
apt-get update
apt-get install -y build-essential cmake libopencv-dev
```

REGULUS 크로스 컴파일에서는 [C++ 런타임 가이드](../README.KR.md)에 설명된 Mobilint 툴체인을 활성화하세요.

## 개요

`infer_depth.cc`의 런타임 흐름은 다음 단계를 수행합니다.

1. MXQ 모델을 로드하고 실행합니다.
2. 입력 이미지를 읽고 `768x768` YOLO 방식 letterbox 전처리를 적용합니다.
3. 모델이 알려주는 HWC 또는 CHW 레이아웃에 맞춰 `uint8` RGB tensor를 구성합니다.
4. Mobilint NPU에서 추론을 실행합니다.
5. 1/4 크기의 MXQ depth 출력을 4배 upsampling합니다.
6. Letterbox 패딩을 제거하고 depth map을 원본 이미지 크기로 조정합니다.
7. Inverse depth를 컬러로 변환해 원본 이미지 위에 합성합니다.

컴파일 튜토리얼에서는 `Uint8InputConfig`를 사용해 `/255` 정규화를 MXQ 모델에 융합합니다. 따라서 이 런타임 예제는 `uint8` 픽셀을 입력합니다.

## 이 튜토리얼의 파일

- `infer_depth.cc`: 모델을 로드하고 추론을 실행한 뒤 결과를 저장합니다.
- `utils/preprocess/`: Letterbox, BGR-to-RGB 변환, 입력 레이아웃 구성을 처리합니다.
- `utils/postprocess/`: ONNX 출력 shape 복원, 패딩 제거, depth 시각화를 처리합니다.
- `CMakeLists.txt`: `infer-depth` 실행 파일을 빌드합니다.

## 필수 MXQ 출력 Upsampling

ONNX 모델은 `(1, 1, 768, 768)`을 출력하지만, MXQ 런타임은 1/4 크기인 `(1, 192, 192)` tensor를 반환합니다. C++ 후처리는 다음 Python 코드와 같은 연산을 수행합니다.

```python
F.interpolate(
    depth,
    scale_factor=4.0,
    mode="bilinear",
    align_corners=False,
)
```

OpenCV linear interpolation으로 `192x192` 출력을 `768x768`로 조정합니다. OpenCV의 half-pixel linear sampling은 PyTorch bilinear interpolation의 `align_corners=False`와 같습니다. 프로그램은 후처리를 계속하기 전에 출력 크기가 정확히 4배 관계인지 확인합니다.

Upsampling이 끝나면 정확한 letterbox 패딩 영역을 제거하고 depth map을 원본 이미지 크기로 복원합니다.

## 빌드

이 디렉토리에서 다음 명령을 실행하세요.

```bash
cmake -B build -S .
cmake --build build -j
```

생성되는 바이너리:

- `build/infer-depth`

다음 명령으로 타겟 아키텍처를 확인할 수 있습니다.

```bash
file build/infer-depth
```

## 실행

실행 파일은 다음 명령줄 형식을 사용합니다.

```bash
./infer-depth <model.mxq> <image_path> <output_path>
```

### ARIES

공용 bus 이미지를 사용해 예제를 실행합니다.

```bash
./build/infer-depth \
  ../../../compilation/depth_estimation/yolo26m-depth.mxq \
  ../../python/rc/bus.jpg \
  ./tmp/bus_depth_demo.jpg
```

### REGULUS

`infer-depth`, `yolo26m-depth.mxq`, `bus.jpg`를 타겟 보드로 복사한 뒤 실행하세요.

```bash
chmod +x infer-depth
./infer-depth yolo26m-depth.mxq bus.jpg bus_depth_demo.jpg
```

## 예상 출력

프로그램은 입력 shape, 원본 이미지 크기, 추론 시간, raw MXQ 출력 shape을 출력합니다. 이후 가까운 영역은 따뜻한 색으로, 먼 영역은 차가운 색으로 표시한 `tmp/bus_depth_demo.jpg` 같은 depth overlay 이미지를 저장합니다.
