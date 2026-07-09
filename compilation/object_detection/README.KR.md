# 객체 탐지 모델 컴파일

이 튜토리얼은 Mobilint `qbcompiler`로 객체 탐지 모델을 컴파일하는 방법을 설명합니다.

예제로는 Ultralytics가 COCO 데이터셋으로 사전 학습한 [YOLO11m](https://docs.ultralytics.com/models/yolo11/)을 사용합니다. 이 모델은 한 이미지 안에서 여러 객체를 탐지하고 위치를 찾을 수 있습니다.

## 사전 준비

시작하기 전에 다음을 준비하세요.

- `qbcompiler`
- COCO 데이터셋에 접근할 수 있는 Hugging Face 계정

필요한 Python 패키지는 다음과 같습니다.

```bash
pip install ultralytics aiohttp aiofiles
```

## 개요

워크플로우는 세 단계로 구성됩니다.

1. **모델 준비**: 모델을 다운로드하고 ONNX로 export합니다.
2. **캘리브레이션 데이터셋 준비**: COCO에서 대표성 있는 캘리브레이션 데이터셋을 만듭니다.
3. **모델 컴파일**: 캘리브레이션 데이터를 사용해 모델을 `.mxq`로 변환합니다.

## 1단계: 모델 준비

`ultralytics`를 사용해 사전 학습된 모델을 다운로드하고 ONNX로 export합니다.

```bash
yolo export model=yolo11m.pt format=onnx
```

명령이 끝나면 현재 디렉토리에 `yolo11m.onnx`가 생성됩니다.

## 2단계: 캘리브레이션 데이터셋 준비

캘리브레이션 데이터셋은 모델의 실제 입력 분포를 잘 대표해야 합니다. YOLO11m은 [COCO 데이터셋](https://cocodataset.org/#download)으로 학습되었으므로, 이 튜토리얼에서는 COCO 샘플을 사용합니다.

데이터를 다운로드하기 전에 Hugging Face 토큰으로 로그인하세요.

```bash
hf auth login --token <your_huggingface_token>
```

토큰을 새로 만들거나 확인해야 한다면 [Hugging Face account settings](https://huggingface.co/settings/tokens)를 참고하세요.

`prepare_coco.py`를 실행하면 무작위로 선택한 COCO 이미지를 `coco-selected` 디렉토리에 다운로드합니다.

```bash
python prepare_coco.py
```

이 스크립트는 다음을 수행합니다.

- Hugging Face에서 COCO 이미지 URL을 가져옵니다.
- 캘리브레이션용 이미지를 무작위로 선택합니다.
- 다운로드한 이미지를 `coco-selected`에 저장합니다.

**출력:**

- `coco-selected`: 캘리브레이션 데이터셋 디렉토리

## 2-1단계 (선택): 이미지를 전처리된 Tensor로 변환

캘리브레이션 데이터를 전처리된 `.npy` 텐서 형태로 직접 준비할 수도 있습니다. 모델에 커스텀 전처리가 필요하고, 텐서 입력을 직접 생성하고 싶을 때 유용합니다.

`qbcompiler` v1.0.0부터는 표준 캘리브레이션 데이터셋 생성 흐름을 보통 그대로 사용하면 되므로, 이 단계는 전처리를 직접 제어해야 할 때만 사용하세요.

변환 스크립트는 다음 조건의 전처리 함수를 가정합니다.

- 입력으로 이미지 경로를 받음
- 반환값이 NumPy 텐서임
- 캘리브레이션용 텐서가 `HWC` 형식임

전처리 함수 예시는 다음과 같습니다.

```python
def pre_ftn(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h0, w0 = img.shape[:2]
    r = min(640 / h0, 640 / w0)
    new_unpad = int(round(w0 * r)), int(round(h0 * r))
    dh, dw = 640 - new_unpad[1], 640 - new_unpad[0]

    dw /= 2
    dh /= 2
    if (img.shape[1], img.shape[0]) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    img = (img / 255).astype(np.float32)

    return img
```

스크립트는 `make_calib_man()`으로 텐서 데이터셋을 생성합니다.

```python
make_calib_man(
    pre_ftn=pre_ftn,
    data_dir=args.source_path,
    save_dir=os.path.dirname(args.npy_path),
    save_name=os.path.basename(args.npy_path),
    remove_npy=True,
)
```

실행 명령은 다음과 같습니다.

```bash
python convert_img_to_tensor.py
```

기본값으로 `./coco-selected`의 이미지를 읽고, `./calib_data_tensor`에 텐서 데이터셋을 저장합니다.

## 3단계: 모델 컴파일

컴파일 전에 필요한 전처리 단계를 먼저 확인하세요. YOLO 모델은 일반적으로 [Ultralytics repository](https://github.com/ultralytics/ultralytics)에서 설명하는 `LetterBox` 연산을 사용합니다.

Mobilint 컴파일 API는 캘리브레이션 중에 전처리 파이프라인을 적용합니다. 정규화 단계인 `/255` 스케일링은 `Uint8InputConfig`를 통해 MXQ 모델에 융합되므로, 런타임 모델은 `uint8` 입력을 직접 받을 수 있습니다. 반면 letterbox 같은 공간 변환은 융합되지 않으므로 런타임에서 계속 적용해야 합니다.

`model_compile.py`에서는 전처리 파이프라인을 다음과 같이 정의합니다.

```python
preprocess_pipeline = [
    {
        "op": "letterbox",
        "height": 640,
        "width": 640,
        "padValue": 114,
    }
]
preprocessing_config = PreprocessingConfig(
    apply=True,
    auto_convert_format=True,
    pipeline=preprocess_pipeline,
    input_configs={},
)
```

전처리 융합을 사용할 때는 MXQ 입력 타입을 `uint8`로 설정합니다.

```python
mxq_compile(
    # ... model, calibration data, backend, and target device settings
    preprocessing_config=preprocessing_config,
    uint8_input_config=Uint8InputConfig(apply=True, inputs=[]),
    calibration_config=calibration_config,
)
```

원래 입력 형식을 유지하려면 전처리 융합과 `Uint8InputConfig`를 모두 비활성화하세요.

예제에서는 다음과 같은 양자화 설정을 사용합니다.

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

설정을 마치면 하드웨어에 맞는 `--target-device`를 지정해 `model_compile.py`를 실행하세요. 한 번 실행하면 양자화된 MXQ 파일(`--save-path`)과 중간 MBLT 그래프(`--mblt-path`)가 함께 생성됩니다.

## 3-1단계 (선택): 준비한 Tensor 파일로 컴파일

이미 `.npy` 텐서 파일을 준비했다면, raw 이미지와 전처리 파이프라인 대신 해당 디렉토리를 `calib_data_path`로 사용할 수 있습니다.

```python
mxq_compile(
    model=args.onnx_path,
    calib_data_path=args.calib_data_path,
    save_path=args.save_path,
    image_channels=3,
    backend="onnx",
    device="gpu",
    target_device=args.target_device,
    inference_scheme=inferece_sheme,
    calibration_config=calibration_config,
)
```

파라미터:

- `--onnx-path`: ONNX 모델 경로
- `--calib-data-path`: 캘리브레이션 데이터 경로
- `--save-path`: MXQ 모델 저장 경로 (`onnx -> mxq`)
- `--mblt-path`: MBLT 중간 그래프 저장 경로 (`onnx -> mblt`)
- `--target-device`: 대상 NPU. 아래 표를 참고하세요. inference scheme은 자동으로 선택됩니다 (`ARIES = all`, `REGULUS = single`).

출력:

- `--save-path`에 저장되는 MXQ 모델
- `--mblt-path`에 저장되는 MBLT 중간 그래프

### 대상 디바이스 선택 (`--target-device`)

권장 모델은 대상 디바이스에 따라 다릅니다.

- `regulus-ra`는 YOLOv9 이하를 지원하므로 `yolov9m`을 사용합니다.
- `aries-rb`와 `regulus-rb`는 `yolo11m`을 사용합니다.

| 사용자 | `--target-device` | 모델 |
| --- | --- | --- |
| ARIES | `aries-rb` | `yolo11m` |
| REGULUS (2026-06 이전 고객) | `regulus-ra` | `yolov9m` |
| REGULUS (2026-06 이후 고객) | `regulus-rb` | `yolo11m` |

먼저 디바이스에 맞는 모델을 export하세요. 1단계는 `yolo11m` 기준이며, `regulus-ra`는 대신 `yolov9m`을 export해야 합니다.

```bash
# YOLO11 (aries-rb / regulus-rb용)
yolo export model=yolo11m.pt format=onnx

# YOLOv9 (regulus-ra용)
yolo export model=yolov9m.pt format=onnx
```

그다음 컴파일을 실행합니다.

```bash
# ARIES
python model_compile.py --onnx-path ./yolo11m.onnx --calib-data-path ./coco-selected --save-path ./yolo11m.mxq --mblt-path ./yolo11m.mblt --target-device aries-rb

# REGULUS (2026-06 이전 고객)
python model_compile.py --onnx-path ./yolov9m.onnx --calib-data-path ./coco-selected --save-path ./yolov9m.mxq --mblt-path ./yolov9m.mblt --target-device regulus-ra

# REGULUS (2026-06 이후 고객)
python model_compile.py --onnx-path ./yolo11m.onnx --calib-data-path ./coco-selected --save-path ./yolo11m.mxq --mblt-path ./yolo11m.mblt --target-device regulus-rb
```

명령이 끝나면 해당 MXQ와 MBLT 파일이 현재 디렉토리에 저장됩니다.
