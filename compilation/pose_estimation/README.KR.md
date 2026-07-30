# 포즈 추정 모델 컴파일

이 튜토리얼은 Mobilint `qbcompiler`로 포즈 추정 모델을 컴파일하는 방법을 설명합니다.

예제에서는 Ultralytics가 COCO 데이터셋으로 사전 학습한 [YOLO11m-pose](https://docs.ultralytics.com/models/yolo11/) 모델을 사용합니다. 이 모델은 이미지 속 객체의 골격 키포인트를 추정합니다.

## 사전 준비

시작하기 전에 다음 항목이 준비되어 있는지 확인하세요.

- `qbcompiler`
- gated COCO 데이터셋에 접근할 수 있는 Hugging Face 계정

필요한 Python 패키지는 아래와 같이 설치합니다.

```bash
pip install ultralytics aiohttp aiofiles
```

## 개요

컴파일 워크플로우는 다음 세 단계로 구성됩니다.

1. **모델 준비**: 모델을 다운로드하고 ONNX로 export합니다.
2. **캘리브레이션 데이터셋 준비**: COCO에서 대표성 있는 캘리브레이션 데이터셋을 만듭니다.
3. **모델 컴파일**: 캘리브레이션 데이터를 사용해 ONNX 모델을 `.mxq`로 변환합니다.

## 1단계: 모델 준비

`ultralytics` 패키지를 사용해 사전 학습된 모델을 다운로드하고 ONNX로 export합니다.

```bash
yolo export model=yolo11m-pose.pt format=onnx
```

명령이 끝나면 export된 모델이 현재 디렉터리에 `yolo11m-pose.onnx`로 저장됩니다.

## 2단계: 캘리브레이션 데이터셋 준비

캘리브레이션 데이터셋은 모델이 실제로 받게 될 입력 분포를 잘 대표해야 합니다. YOLO11m-pose는 [COCO 데이터셋](https://cocodataset.org/#download)으로 학습되었기 때문에, 이 튜토리얼에서도 COCO 샘플을 캘리브레이션에 사용합니다.

데이터셋에 접근하기 전에 [Hugging Face](https://huggingface.co/)에 로그인하고 토큰으로 인증하세요.

```bash
hf auth login --token <your_huggingface_token>
```

토큰을 모를 경우 [Hugging Face 계정 설정](https://huggingface.co/settings/tokens)에서 확인할 수 있습니다.

`prepare_coco.py`를 실행하면 데이터셋 준비 과정을 자동화할 수 있습니다. 이 스크립트는 COCO 이미지 URL을 읽고, 샘플을 무작위로 선택한 뒤, `coco-selected` 디렉터리에 다운로드합니다.

```bash
python prepare_coco.py
```

**이 스크립트가 수행하는 작업:**

- Hugging Face에서 COCO 이미지 URL 다운로드
- 캘리브레이션용 이미지 무작위 선택
- 선택한 이미지를 `coco-selected`에 저장

**출력:**

- `coco-selected`: 캘리브레이션 데이터셋 디렉터리

`coco-selected`에 저장된 이미지들을 캘리브레이션 데이터셋으로 사용합니다.

## 2-1단계 (선택): 이미지를 전처리된 텐서로 변환

캘리브레이션 데이터셋을 전처리된 `.npy` 텐서 형태로 준비할 수도 있습니다. 모델에 커스텀 전처리가 필요하고, 캘리브레이션 입력을 직접 만들고 싶을 때 유용합니다.

`qbcompiler` v1.0.0부터는 일반적인 이미지 기반 캘리브레이션만으로 충분한 경우가 많습니다. 이 단계는 전처리를 직접 제어해야 할 때만 사용하세요.

변환 스크립트는 다음 조건을 만족하는 전처리 함수를 가정합니다.

- 입력으로 이미지 경로를 받음
- 출력으로 NumPy 텐서를 반환함
- 캘리브레이션용 `HWC` 형식 텐서를 생성함

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

스크립트는 `make_calib_man()`을 사용해 텐서 데이터셋을 생성합니다.

```python
make_calib_man(
    pre_ftn=pre_ftn,
    data_dir=args.source_path,
    save_dir=os.path.dirname(args.npy_path),
    save_name=os.path.basename(args.npy_path),
    remove_npy=True,
)
```

스크립트 실행 방법은 다음과 같습니다.

```bash
python convert_img_to_tensor.py
```

기본값으로 `./coco-selected`에서 이미지를 읽고, `./calib_data_tensor`에 텐서 데이터셋을 저장합니다.

## 3단계: 모델 컴파일

컴파일 전에 필요한 전처리 요구사항을 확인하세요. YOLO 모델은 일반적으로 [Ultralytics repository](https://github.com/ultralytics/ultralytics)에 설명된 `LetterBox` 연산을 사용합니다.

Mobilint 컴파일 API는 캘리브레이션 과정에서 전처리 파이프라인을 적용합니다. 정규화 단계(`/255` 스케일링)는 `Uint8InputConfig`를 통해 MXQ 모델에 융합되므로, 런타임 모델은 `uint8` 입력을 직접 받을 수 있습니다. 반면 letterboxing 같은 공간 변환은 융합되지 않으므로 런타임에서 계속 적용해야 합니다.

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

원래 입력 형식을 유지하고 싶다면 전처리 융합과 `Uint8InputConfig`를 모두 비활성화하세요.

예제에서는 다음 양자화 설정을 사용합니다.

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

설정을 마친 뒤, 하드웨어에 맞는 `--target-device`를 지정해 `model_compile.py`를 실행합니다. 한 번 실행하면 양자화된 MXQ 파일(`--save-path`)과 중간 MBLT 그래프(`--mblt-path`)가 함께 생성됩니다.

MXQ 컴파일 시 `model_compile.py`는 `torch.cuda.is_available()`이 참이면 CUDA를 사용하고, 그렇지 않으면 CPU로 자동 전환합니다. 따라서 GPU 지원 이미지와 CPU 전용 `qbcompiler` 이미지에서 모두 실행할 수 있으며, 선택된 호스트 디바이스는 컴파일 시작 전에 출력됩니다.

## 3-1단계 (선택): 준비된 텐서 파일로 컴파일

이미 `.npy` 텐서 파일을 준비했다면, 원본 이미지와 전처리 파이프라인 대신 해당 디렉터리를 `calib_data_path`로 사용할 수 있습니다.

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

**파라미터:**

- `--onnx-path`: ONNX 모델 경로
- `--calib-data-path`: 캘리브레이션 데이터 경로
- `--save-path`: MXQ 모델 저장 경로 (`onnx -> mxq`)
- `--mblt-path`: MBLT 중간 그래프 저장 경로 (`onnx -> mblt`)
- `--target-device` (필수): 대상 NPU. 아래 표를 참고하세요. inference scheme은 자동으로 선택됩니다 (`ARIES = all`, `REGULUS = single`).

**출력:**

- `--save-path`에 저장되는 MXQ 모델 (`onnx -> mxq`, 양자화된 NPU 패키지)
- `--mblt-path`에 저장되는 MBLT 중간 그래프 (`onnx -> mblt`, 양자화 전 그래프)

### 대상 디바이스 선택 (`--target-device`)

필요한 모델은 대상 디바이스에 따라 달라집니다. 구형 REGULUS 하드웨어(`regulus-ra`, 2026-06 이전 고객)는 YOLOv9 이하만 지원하므로 `yolov8m-pose`를 사용합니다. ARIES(`aries-rb`)와 신형 REGULUS 하드웨어(`regulus-rb`, 2026-06 이후 고객)는 `yolo11m-pose`를 사용합니다.

| 사용자 | `--target-device` | 모델 |
| --- | --- | --- |
| ARIES | `aries-rb` | `yolo11m-pose` |
| REGULUS (2026-06 이전 고객) | `regulus-ra` | `yolov8m-pose` |
| REGULUS (2026-06 이후 고객) | `regulus-rb` | `yolo11m-pose` |

먼저 대상 디바이스에 맞는 모델을 export하세요. 1단계는 `yolo11m-pose` 기준이며, `regulus-ra`를 사용할 경우 `yolov8m-pose`를 export해야 합니다.

```bash
# YOLO11 pose (aries-rb / regulus-rb용)
yolo export model=yolo11m-pose.pt format=onnx

# YOLOv8 pose (regulus-ra용)
yolo export model=yolov8m-pose.pt format=onnx
```

```bash
# ARIES
python model_compile.py --onnx-path ./yolo11m-pose.onnx --calib-data-path ./coco-selected --save-path ./yolo11m-pose.mxq --mblt-path ./yolo11m-pose.mblt --target-device aries-rb

# REGULUS (2026-06 이전 고객)
python model_compile.py --onnx-path ./yolov8m-pose.onnx --calib-data-path ./coco-selected --save-path ./yolov8m-pose.mxq --mblt-path ./yolov8m-pose.mblt --target-device regulus-ra

# REGULUS (2026-06 이후 고객)
python model_compile.py --onnx-path ./yolo11m-pose.onnx --calib-data-path ./coco-selected --save-path ./yolo11m-pose.mxq --mblt-path ./yolo11m-pose.mblt --target-device regulus-rb
```
