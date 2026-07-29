# C++ 시맨틱 세그멘테이션 런타임

이 튜토리얼은 C++ `qbruntime` API로 컴파일된 `YOLO26m-sem` MXQ 모델을 실행하고 Cityscapes 시맨틱 세그멘테이션 overlay 이미지를 저장하는 방법을 설명합니다.

시작하기 전에 [시맨틱 세그멘테이션 컴파일 튜토리얼](../../../compilation/semantic_segmentation/README.KR.md)을 완료하세요. 아래 예제는 `../../../compilation/semantic_segmentation/yolo26m-sem.mxq`를 사용합니다.

## 사전 준비

다음 구성 요소가 준비되어 있어야 합니다.

- Mobilint NPU 드라이버 및 C++ `qbruntime`
- OpenCV 개발 라이브러리
- C++17 컴파일러
- CMake `3.21` 이상
- 컴파일된 `yolo26m-sem.mxq` 모델

Ubuntu 또는 Debian 기반 ARIES 네이티브 빌드에서는 다음과 같이 설치할 수 있습니다.

```bash
apt-get update
apt-get install -y build-essential cmake libopencv-dev
```

REGULUS 크로스 컴파일에서는 [C++ 런타임 가이드](../README.KR.md)에 설명된 Mobilint 툴체인을 활성화하세요.

## 개요

`infer_semantic.cc`의 런타임 흐름은 다음 단계를 수행합니다.

1. MXQ 모델을 불러오고 실행합니다.
2. `munster.png`를 읽고 `1024x2048` YOLO 방식 letterbox 전처리를 적용합니다.
3. 모델이 알려주는 HWC 또는 CHW 레이아웃에 맞춰 `uint8` RGB tensor를 구성합니다.
4. Mobilint NPU에서 추론을 실행합니다.
5. 19개 Cityscapes logit에 `argmax`를 적용합니다.
6. Letterbox 패딩을 제거하고 class map을 원본 이미지 크기로 복원합니다.
7. 공식 Cityscapes palette를 적용하고 원본 이미지 위에 결과를 합성합니다.

컴파일 튜토리얼에서는 `Uint8InputConfig`를 사용해 `/255` 정규화를 MXQ 모델에 융합합니다. 따라서 이 런타임 예제는 `uint8` 픽셀을 입력합니다.

## 이 튜토리얼의 파일

- `infer_semantic.cc`: 모델을 불러오고 추론을 실행한 뒤 결과를 저장합니다.
- `utils/preprocess/`: Letterbox, BGR-to-RGB 변환, 입력 레이아웃 구성을 처리합니다.
- `utils/postprocess/`: `argmax`, class map 복원, Cityscapes overlay 렌더링을 처리합니다.
- `CMakeLists.txt`: `infer-semantic` 실행 파일을 빌드합니다.

## 입출력 Shape

컴파일된 모델이 알려주는 shape은 다음과 같습니다.

- 입력: `(1024, 2048, 3)` HWC `uint8`
- 출력: `(1024, 2048, 19)` HWC `float32` logit

ONNX 모델은 graph 안에 `argmax`가 포함되어 `(1, 1024, 2048)` class map을 반환하지만, MXQ 모델은 19개 logit을 모두 출력합니다. C++ 후처리는 각 픽셀에서 점수가 가장 높은 class를 계산합니다. HWC와 CHW 출력 레이아웃을 모두 지원합니다.

MXQ 출력의 공간 크기는 ONNX class map의 공간 크기와 같으므로 별도의 output upsampling은 필요하지 않습니다. 원본 이미지 크기가 다를 때 class map을 복원하는 과정에서만 nearest-neighbor interpolation을 사용합니다.

제공된 `munster.png`는 이미 `2048x1024` 크기이므로 기본 모델에서는 letterbox 패딩을 추가하지 않습니다.

## 빌드

이 디렉토리에서 다음 명령을 실행하세요.

```bash
cmake -B build -S .
cmake --build build -j
```

생성되는 바이너리:

- `build/infer-semantic`

다음 명령으로 타겟 아키텍처를 확인할 수 있습니다.

```bash
file build/infer-semantic
```

## 실행

실행 파일은 다음 명령줄 형식을 사용합니다.

```bash
./infer-semantic <model.mxq> <image_path> <output_path>
```

### ARIES

공용 Münster 이미지를 사용해 예제를 실행합니다.

```bash
./build/infer-semantic \
  ../../../compilation/semantic_segmentation/yolo26m-sem.mxq \
  ../../python/rc/munster.png \
  ./tmp/munster_semantic_demo.png
```

### REGULUS

`infer-semantic`, `yolo26m-sem.mxq`, `munster.png`를 타겟 보드로 복사한 뒤 실행하세요.

```bash
chmod +x infer-semantic
./infer-semantic yolo26m-sem.mxq munster.png munster_semantic_demo.png
```

## 예상 출력

프로그램은 입력 shape, 원본 이미지 크기, 추론 시간, raw MXQ 출력 shape을 출력합니다. 이후 각 예측 class를 공식 Cityscapes 색상으로 표시한 `2048x1024` 크기의 `tmp/munster_semantic_demo.png`를 저장합니다.
