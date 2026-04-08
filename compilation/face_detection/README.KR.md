# 얼굴 탐지 모델 컴파일

이 튜토리얼은 Mobilint `qbcompiler`를 사용해 얼굴 탐지 모델을 컴파일하는 방법을 설명합니다.

전체 흐름은 [../object_detection/README.KR.md](../object_detection/README.KR.md)와 비슷하게 구성되어 있습니다.

1. 사전 학습된 모델을 준비하고 ONNX로 export합니다.
2. 대표성 있는 캘리브레이션 데이터셋을 구성합니다.
3. 모델을 Mobilint `.mxq` 형식으로 컴파일합니다.

이 예제에서는 `yolo-face` 프로젝트의 [YOLOv12m-face](https://github.com/akanametov/yolo-face) 모델을 사용합니다. 이 모델은 얼굴 바운딩 박스를 검출하는 단일 클래스 탐지기이며, `640x640` 입력과 letterbox 전처리를 사용합니다.

## 사전 준비

시작하기 전에 다음이 준비되어 있어야 합니다.

- `qbcompiler`
- Python 패키지: `ultralytics`, `huggingface_hub`

필요한 Python 패키지는 아래 명령으로 설치할 수 있습니다.

```bash
pip install ultralytics huggingface_hub
```

환경에 따라 Hugging Face 인증이 필요하다면, 캘리브레이션 데이터셋을 내려받기 전에 아래와 같이 로그인하세요.

```bash
hf auth login --token <your_huggingface_token>
```

## 개요

얼굴 탐지 컴파일 워크플로우는 세 단계로 구성됩니다.

1. **모델 준비**: 사전 학습된 얼굴 탐지 모델을 다운로드하고 ONNX로 export합니다.
2. **캘리브레이션 데이터셋 준비**: WIDER FACE에서 작지만 대표성 있는 캘리브레이션 셋을 만듭니다.
3. **모델 컴파일**: 선택한 이미지를 사용해 ONNX 모델을 `.mxq`로 컴파일합니다.

## 1단계: 모델 준비

`prepare_model.py`를 실행해 사전 학습된 YOLO 얼굴 탐지 가중치를 다운로드하고 ONNX로 export합니다.

```bash
python prepare_model.py
```

**수행 작업:**

- 파일이 없으면 upstream release에서 `yolov12m-face.pt`를 다운로드합니다.
- `ultralytics.YOLO`로 가중치를 로드합니다.
- 모델을 `yolov12m-face.onnx`로 export합니다.

**출력:**

- `yolov12m-face.pt`
- `yolov12m-face.onnx`

## 2단계: 캘리브레이션 데이터셋 준비

객체 탐지 튜토리얼과 마찬가지로, 캘리브레이션 데이터는 실제 배포 시 입력 분포를 잘 대표해야 합니다. 얼굴 탐지 예제에서는 Hugging Face에 공개된 [WIDER FACE](https://huggingface.co/datasets/CUHK-CSE/wider_face) 학습 아카이브를 사용합니다.

데이터셋 준비 스크립트는 다음과 같이 실행합니다.

```bash
python prepare_widerface.py
```

이 스크립트는 `WIDER_train.zip`을 다운로드하고, 학습 이미지를 하위 카테고리별로 묶은 뒤, 각 카테고리에서 무작위로 한 장씩 선택해 `widerface-selected/`에 저장합니다.

출력 디렉토리와 랜덤 시드를 직접 지정할 수도 있습니다.

```bash
python prepare_widerface.py --output-dir ./widerface-selected --seed 42
```

**수행 작업:**

- Hugging Face에서 `WIDER_train.zip`을 다운로드합니다.
- `WIDER_train/images` 아래 이미지를 읽습니다.
- WIDER FACE 하위 카테고리별로 이미지를 그룹화합니다.
- 각 하위 카테고리에서 무작위로 한 장을 선택합니다.
- 선택한 이미지를 `widerface-selected/`에 저장합니다.

**출력:**

- `widerface-selected/`: 컴파일에 사용할 캘리브레이션 데이터셋

## 3단계: 모델 컴파일

컴파일 전에 모델이 요구하는 전처리를 확인해야 합니다. 객체 탐지 YOLO 예제와 마찬가지로, 이 튜토리얼도 입력 비율을 유지하면서 `640x640` 크기에 맞추기 위해 letterbox 리사이즈를 사용합니다.

`model_compile.py`에서는 다음과 같이 전처리 파이프라인을 정의합니다.

```python
preprocess_pipeline = [{"op": "letterbox", "height": 640, "width": 640, "padValue": 114}]

preprocessing_config = PreprocessingConfig(
    apply=True,
    auto_convert_format=True,
    pipeline=preprocess_pipeline,
    input_configs={},
)
```

스크립트는 `Uint8` 입력 설정도 함께 사용하며, 캘리브레이션 설정은 다음과 같습니다.

```python
calibration_config = CalibrationConfig(
    method=1,  # 0 for per tensor, 1 for per channel
    output=1,  # 0 for layer, 1 for channel
    mode=1,  # maxpercentile
    max_percentile={
        "percentile": 0.9999,
        "topk_ratio": 0.01,
    },
)
```

다음 명령으로 ONNX 모델과 캘리브레이션 데이터셋을 사용해 컴파일합니다.

```bash
python model_compile.py --onnx-path ./yolov12m-face.onnx --calib-data-path ./widerface-selected --save-path ./yolov12m-face.mxq
```

이 예제는 `prepare_model.py`의 출력 파일명과 맞추기 위해 경로를 명시적으로 전달합니다.

**수행 작업:**

- ONNX 모델을 로드합니다.
- 캘리브레이션 이미지를 로드합니다.
- 모델을 `.mxq` 형식으로 컴파일합니다.
- ONNX 파일 옆에 중간 산출물인 `.mblt` 그래프도 저장합니다.

**파라미터:**

- `--onnx-path`: ONNX 모델 파일 경로
- `--calib-data-path`: 캘리브레이션 이미지 디렉토리 경로
- `--save-path`: 컴파일된 `.mxq` 모델을 저장할 경로

**출력:**

- `yolov12m-face.mxq`
- `yolov12m-face.mblt`

명령이 끝나면 [../../runtime/face_detection/README.KR.md](../../runtime/face_detection/README.KR.md)에서 컴파일된 모델로 추론을 실행할 수 있습니다.
