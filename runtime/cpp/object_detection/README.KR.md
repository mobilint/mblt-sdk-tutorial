# C++ 객체 탐지 런타임

이 튜토리얼은 C++ `qbruntime` API로 컴파일된 YOLO 객체 탐지 MXQ 모델을 실행하는 방법을 설명합니다.

시작하기 전에 [../../../compilation/object_detection/README.KR.md](../../../compilation/object_detection/README.KR.md)의 컴파일 과정을 먼저 완료하세요. 이 런타임 예제는 다음 모델 중 하나를 사용합니다.

- ARIES: `yolo11m.mxq`
- REGULUS: `yolov9m.mxq`

## 사전 준비

다음 구성 요소가 준비되어 있어야 합니다.

- Mobilint `qbruntime`
- OpenCV 개발 라이브러리
- C++17 컴파일러
- CMake `3.21` 이상
- 컴파일 튜토리얼에서 생성한 해당 MXQ 파일

Ubuntu 또는 Debian 기반 ARIES 네이티브 빌드에서는 다음과 같이 설치할 수 있습니다.

```bash
apt-get update
apt-get install -y build-essential cmake libopencv-dev
```

REGULUS 크로스 컴파일에서는 먼저 [../README.KR.md](../README.KR.md)에 정리된 Mobilint 툴체인을 활성화하세요.

## 개요

`infer_det.cc`의 런타임 흐름은 다음 단계를 따릅니다.

1. 컴파일된 MXQ 모델을 로드합니다.
2. 입력 이미지를 읽습니다.
3. `Transformer`를 통해 YOLO 스타일 letterbox 전처리를 적용합니다.
4. Mobilint NPU에서 추론을 실행합니다.
5. anchor-free YOLO 출력을 DFL과 NMS로 디코드합니다.
6. 탐지 결과를 원본 이미지 좌표로 되돌린 뒤 시각화합니다.

프로그램은 `uint8` 입력을 사용하며, 정규화는 컴파일된 모델에 이미 융합되어 있다고 가정합니다.

## 이 튜토리얼의 파일

- `infer_det.cc`: 전체 객체 탐지 파이프라인을 실행하고 결과 이미지를 저장합니다.
- `yolo_detect_config.h`: detect head 설정, 임계값, 입력 이미지 크기를 정의합니다.
- `utils/inference/`: 모델 실행과 전처리에 사용하는 공용 런타임 유틸리티입니다.
- `utils/postprocess/`: 디코드와 NMS에 사용하는 공용 유틸리티입니다.
- `CMakeLists.txt`: `infer-det` 실행 파일과 보조 유틸리티 라이브러리를 빌드합니다.

## 프로그램 동작 방식

프로그램은 다음 명령줄 형식을 사용합니다.

```bash
./infer-det <model.mxq> <image_path> <output_path>
```

`Transformer`는 다음 작업을 처리합니다.

- Letterbox 리사이즈
- BGR에서 RGB로 변환
- HWC에서 CHW로 변환

추론 후에는 `YoloDecoder`가 DFL 디코드, confidence filtering, NMS, 좌표 복원을 수행하고, 최종 결과를 COCO 클래스 라벨과 함께 그립니다.

## 빌드

이 디렉토리에서 다음 명령을 실행하세요.

```bash
cmake -B build -S .
cmake --build build -j
```

생성되는 바이너리:

- `build/infer-det`

다음 명령으로 타겟 아키텍처를 확인할 수 있습니다.

```bash
file build/infer-det
```

## 실행

샘플 이미지:

- `../rc/cr7.jpg`

### ARIES

```bash
./build/infer-det ../../../compilation/object_detection/yolo11m.mxq ../rc/cr7.jpg result.jpg
```

### REGULUS

`build/infer-det`, `yolov9m.mxq`, `cr7.jpg`를 타겟 보드로 복사한 뒤 다음 명령을 실행하세요.

```bash
chmod +x infer-det
./infer-det yolov9m.mxq cr7.jpg result.jpg
```

## 예상 출력

프로그램은 모델 입력 shape, 원본 이미지 크기, 추론 시간, 디코드된 탐지 결과를 출력한 뒤, 바운딩 박스와 클래스 라벨이 포함된 `result.jpg` 같은 결과 이미지를 저장합니다.
