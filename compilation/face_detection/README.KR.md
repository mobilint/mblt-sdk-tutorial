# 얼굴 탐지 모델 컴파일

이 튜토리얼은 Mobilint `qbcompiler`로 얼굴 탐지 모델을 컴파일하는 방법을 설명합니다.

전체 워크플로는 [../object_detection/README.KR.md](../object_detection/README.KR.md)와 유사합니다.

1. 사전 학습된 모델을 준비하고 ONNX로 내보냅니다.
2. 대표성 있는 캘리브레이션 데이터셋을 구성합니다.
3. 모델을 Mobilint `.mxq` 형식으로 컴파일합니다.

이 예제에서는 `yolo-face` 프로젝트의 [YOLOv12m-face](https://github.com/akanametov/yolo-face) 모델을 사용합니다. 이 모델은 얼굴 바운딩 박스를 검출하는 단일 클래스 탐지기이며, `640x640` 입력과 letterbox 전처리를 사용합니다.

## 사전 준비

시작하기 전에 다음 항목이 준비되어 있어야 합니다.

- `qbcompiler`
- Python 패키지: `ultralytics`, `huggingface_hub`

필요한 Python 패키지는 아래 명령으로 설치할 수 있습니다.

```bash
pip install ultralytics huggingface_hub
```

환경에 따라 Hugging Face 인증이 필요하다면, 캘리브레이션 데이터셋을 다운로드하기 전에 아래와 같이 로그인하세요.

```bash
hf auth login --token <your_huggingface_token>
```

## 개요

얼굴 탐지 컴파일 워크플로우는 세 단계로 구성됩니다.

1. **모델 준비**: 사전 학습된 얼굴 탐지 모델을 다운로드하고 ONNX로 내보냅니다.
2. **캘리브레이션 데이터셋 준비**: WIDER FACE에서 작지만 대표성 있는 캘리브레이션 셋을 만듭니다.
3. **모델 컴파일**: 선택한 이미지를 사용해 ONNX 모델을 `.mxq`로 컴파일합니다.

## 1단계: 모델 준비

`prepare_model.py`를 실행해 사전 학습된 YOLO 얼굴 탐지 가중치를 다운로드하고 ONNX로 내보냅니다.

```bash
python prepare_model.py
```

**수행 작업:**

- 로컬에 파일이 없으면 upstream release에서 `yolov12m-face.pt`를 다운로드합니다.
- `ultralytics.YOLO`로 가중치를 로드합니다.
- 모델을 `yolov12m-face.onnx`로 export합니다.

**출력:**

- `yolov12m-face.pt`
- `yolov12m-face.onnx`

## 2단계: 캘리브레이션 데이터셋 준비

객체 탐지 튜토리얼과 마찬가지로, 캘리브레이션 데이터는 실제 배포 시 예상되는 입력 분포를 잘 대표해야 합니다. 얼굴 탐지 예제에서는 Hugging Face에 공개된 [WIDER FACE](https://huggingface.co/datasets/CUHK-CSE/wider_face) 학습 아카이브를 사용합니다.

데이터셋 준비 스크립트는 다음과 같이 실행합니다.

```bash
python prepare_widerface.py
```

이 스크립트는 `WIDER_train.zip`을 다운로드하고, 학습 이미지를 하위 카테고리별로 묶은 뒤, 각 카테고리에서 무작위로 한 장씩 선택해 `widerface-selected/`에 복사합니다.

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

## 2-1단계 (선택): 이미지를 전처리된 텐서로 변환

캘리브레이션 데이터는 전처리된 `.npy` 텐서 형태로도 준비할 수 있습니다. 모델에 맞는 사용자 정의 전처리 함수를 적용해야 하거나, 텐서 입력을 직접 생성하고 싶을 때 유용합니다.

`qbcompiler` v1.0.0부터는 표준화된 캘리브레이션 데이터셋 생성 흐름을 사용할 수 있으므로, 대부분의 경우 이 단계는 생략해도 됩니다. 전처리를 직접 제어해야 할 때만 사용하세요.

> **중요**: 이 튜토리얼의 현재 `model_compile.py` 흐름은 여전히 `--calib-data-path`로 원본 이미지 디렉토리를 기대하며, 아래에 나온 내장 letterbox 전처리를 항상 적용합니다. 따라서 여기서 생성한 `.npy` 텐서는 이 튜토리얼에서는 전처리 예시용일 뿐이며, 문서에 나온 `model_compile.py` 명령 경로에서는 **직접 사용되지 않습니다**.

변환 스크립트는 다음 조건을 만족하는 전처리 함수를 가정합니다.

- 입력으로 이미지 경로를 받음
- NumPy 텐서를 반환함
- 캘리브레이션 데이터용 텐서를 `HWC` 형식으로 생성함

전처리 함수 예시는 다음과 같습니다.

```python
def pre_ftn(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h0, w0 = img.shape[:2]  # original hw
    r = min(640 / h0, 640 / w0)  # ratio
    new_unpad = int(round(w0 * r)), int(round(h0 * r))
    dh, dw = (
        640 - new_unpad[1],
        640 - new_unpad[0],
    )  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if (img.shape[1], img.shape[0]) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # add border
    img = (img / 255).astype(np.float32)

    return img
```

스크립트는 `make_calib_man()`을 사용해 텐서 데이터셋을 생성합니다.

```python
make_calib_man(
    pre_ftn=pre_ftn,
    data_dir=args.source_path,
    save_dir=os.path.dirname(args.npy_path),
    save_name=os.path.basename(args.npy_path),
    remove_npy=True,  # 새 .npy 파일을 쓰기 전에 기존 출력물을 정리합니다.
)
```

스크립트 실행 방법은 다음과 같습니다.

```bash
python convert_img_to_tensor.py
```

기본 설정에서는 `./widerface-selected`에서 이미지를 읽고, `./calib_data_tensor` 아래에 텐서 데이터셋을 생성합니다.

## 3단계: 모델 컴파일

컴파일 전에 모델이 요구하는 전처리를 확인해야 합니다. YOLO 객체 탐지 예제와 마찬가지로, 이 튜토리얼도 종횡비를 유지하면서 `640x640` 입력 크기에 맞추기 위해 letterbox 리사이즈를 사용합니다.

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

정규화 과정에서 `letterbox` 연산에는 `1/255` 스케일링이 포함되며, 이 전처리는 `fuseIntoFirstLayer`와 `Uint8InputConfig`를 통해 MXQ 모델 안으로 fuse할 수 있습니다.

전처리 fuse를 사용할 때는 MXQ 입력 타입을 `uint8`로 설정해야 합니다.

```python
# ONNX -> MXQ: quantized package that runs on the NPU
mxq_compile(
    # ... model, calibration data, backend, and target device settings
    preprocessing_config=preprocessing_config,
    uint8_input_config=Uint8InputConfig(apply=True, inputs=[]),
    calibration_config=calibration_config,
)
```

원래 입력 형식을 유지하고 싶다면 `fuseIntoFirstLayer`와 `Uint8InputConfig`를 모두 비활성화하세요.

이 예제는 아래와 같은 양자화 설정도 사용합니다.

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

**파라미터:**

- `--onnx-path`: ONNX 모델 파일 경로
- `--calib-data-path`: 캘리브레이션 이미지 디렉토리 경로. 문서에 나온 CLI는 원본 이미지를 입력으로 받고, 컴파일 중에 내장 전처리 파이프라인을 적용합니다.
- `--save-path`: MXQ 모델을 저장할 경로 (onnx -> mxq 산출물)
- `--mblt-path`: MBLT 중간 그래프를 저장할 경로 (onnx -> mblt 산출물)
- `--target-device` (필수): 대상 NPU. 아래 표를 참고하세요. 디바이스에 따라 inference scheme이 자동으로 결정됩니다 (ARIES = `all`, REGULUS = `single`).

**출력:**

- `yolov12m-face.mxq` (onnx -> mxq, 양자화 NPU 패키지)
- `yolov12m-face.mblt` (onnx -> mblt, 양자화 전 그래프)

### 대상 디바이스 선택 (`--target-device`)

| 사용자 | `--target-device` |
| --- | --- |
| ARIES | `aries-rb` |
| REGULUS (2026-06 이후 고객) | `regulus-rb` |

> **참고**: Face detection은 YOLOv12 `yolo-face` 모델을 사용하므로, **구형 REGULUS(`regulus-ra`, 2026-06 이전 고객용)에서는 지원되지 않습니다**. 해당 세대는 YOLOv9 이하만 지원합니다. `aries-rb` 또는 `regulus-rb`를 사용하세요.

```bash
# ARIES
python model_compile.py --onnx-path ./yolov12m-face.onnx --calib-data-path ./widerface-selected --save-path ./yolov12m-face.mxq --mblt-path ./yolov12m-face.mblt --target-device aries-rb

# REGULUS (2026-06 이후 고객)
python model_compile.py --onnx-path ./yolov12m-face.onnx --calib-data-path ./widerface-selected --save-path ./yolov12m-face.mxq --mblt-path ./yolov12m-face.mblt --target-device regulus-rb
```

위 명령을 실행하면 현재 디렉토리에 MXQ 파일(`yolov12m-face.mxq`)과 MBLT 파일(`yolov12m-face.mblt`)이 저장됩니다.

명령이 끝나면 [../../runtime/python/face_detection/README.KR.md](../../runtime/python/face_detection/README.KR.md)에서 컴파일된 모델로 추론을 실행할 수 있습니다.
