# C++ 포즈 추정 런타임

이 튜토리얼은 C++ `qbruntime` API로 컴파일된 YOLO 포즈 추정 MXQ 모델을 실행하는 방법을 설명합니다.

시작하기 전에 [../../../compilation/pose_estimation/README.KR.md](../../../compilation/pose_estimation/README.KR.md)의 컴파일 과정을 먼저 완료하세요. 이 런타임 예제는 다음 모델 중 하나를 사용합니다.

- REGULUS `regulus-rb` (기본): `yolo11m-pose.mxq`
- REGULUS `regulus-ra`: `yolov8m-pose.mxq`
- ARIES `aries-rb`: `yolo11m-pose.mxq`

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

`infer_pose.cc`의 런타임 흐름은 다음 단계를 따릅니다.

1. 컴파일된 MXQ 모델을 로드합니다.
2. 입력 이미지를 읽습니다.
3. `Preprocessor`를 통해 YOLO 스타일 letterbox 전처리를 적용합니다.
4. Mobilint NPU에서 추론을 실행합니다.
5. 박스와 keypoint를 DFL과 NMS로 디코드합니다.
6. 탐지 결과를 원본 이미지 좌표로 되돌립니다.
7. 사람 박스, keypoint, skeleton limb를 그립니다.

`--input-dtype`는 MXQ를 어떻게 컴파일했는지와 일치해야 합니다 ([컴파일 튜토리얼](../../../compilation/pose_estimation/README.KR.md) 참고).

- `uint8`: 정규화를 융합(`Uint8InputConfig`)해 컴파일한 MXQ. letterbox된 원본 픽셀을 그대로 입력합니다.
- `float`: 융합 없이 컴파일한 MXQ. 런타임에서 `/255`를 적용합니다.

플래그가 컴파일한 MXQ와 다르면 결과가 잘못됩니다.

## 이 튜토리얼의 파일

- `infer_pose.cc`: 전체 포즈 추정 파이프라인을 실행하고 결과 이미지를 저장합니다.
- `yolo_pose_config.h`: pose head 설정, 임계값, keypoint 개수, 입력 이미지 크기를 정의합니다.
- `utils/preprocess/`: 전처리 유틸리티(`Preprocessor`)입니다.
- `utils/postprocess/`: 디코드와 NMS에 사용하는 공용 유틸리티입니다.
- `CMakeLists.txt`: `infer-pose` 실행 파일과 보조 유틸리티 라이브러리를 빌드합니다.

## 프로그램 동작 방식

프로그램은 다음 명령줄 형식을 사용합니다.

```bash
./infer-pose <model.mxq> <image_path> <output_path> [--input-dtype uint8|float]
```

`Preprocessor`는 다음 작업을 처리합니다.

- Letterbox 리사이즈
- BGR에서 RGB로 변환
- `Model::infer`용 HWC 버퍼로 패킹

추론 후에는 `YoloPoseDecoder`가 다음 작업을 수행합니다.

- anchor-free YOLO 출력 디코드
- 사람 박스와 keypoint 추출
- confidence filtering과 NMS 적용
- 탐지 결과와 keypoint를 원본 이미지 좌표로 복원

이후 시각화 단계에서 사람 박스, COCO 17 keypoint, skeleton limb를 그립니다.

## 빌드

이 디렉토리에서 다음 명령을 실행하세요.

```bash
cmake -B build -S .
cmake --build build -j
```

생성되는 바이너리:

- `build/infer-pose`

다음 명령으로 타겟 아키텍처를 확인할 수 있습니다.

```bash
file build/infer-pose
```

## 실행

샘플 이미지:

- `../rc/cr7.jpg`

### ARIES

```bash
./build/infer-pose ../../../compilation/pose_estimation/yolo11m-pose.mxq ../rc/cr7.jpg result.jpg
```

### REGULUS (`regulus-rb`)

`build/infer-pose`, `yolo11m-pose.mxq`, `cr7.jpg`를 타겟 보드로 복사한 뒤 다음 명령을 실행하세요.

```bash
chmod +x infer-pose
./infer-pose yolo11m-pose.mxq cr7.jpg result.jpg --input-dtype uint8   # 정규화 융합으로 컴파일한 MXQ
./infer-pose yolo11m-pose.mxq cr7.jpg result.jpg --input-dtype float   # 융합 없이 컴파일한 MXQ
```

## 예상 출력

프로그램은 모델 입력 shape, 원본 이미지 크기, 추론 시간, 디코드된 탐지 결과를 출력한 뒤, 사람 박스, keypoint, skeleton 선이 포함된 `result.jpg` 같은 결과 이미지를 저장합니다.
