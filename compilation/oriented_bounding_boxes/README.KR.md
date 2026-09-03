# 회전 바운딩 박스 모델 컴파일

이 튜토리얼은 Mobilint `qbcompiler`로 회전 바운딩 박스(OBB) 탐지 모델을 컴파일하는 방법을 설명합니다.

이 예제에서는 Ultralytics의 [YOLO11m-obb](https://docs.ultralytics.com/tasks/obb/) 모델을 사용합니다. 이 모델은 DOTA 스타일 항공 영상에서 회전 객체 탐지를 위해 학습되었으므로, 이 튜토리얼에서는 캘리브레이션에 DOTA 데이터셋을 사용하고 `1024x1024` letterbox 전처리를 적용합니다.

## 사전 준비

시작하기 전에 다음 항목이 준비되어 있어야 합니다:

- `qbcompiler`
- Python 3.10 이상

필요한 Python 패키지는 다음과 같이 설치합니다:

```bash
pip install ultralytics opencv-python
```

## 개요

워크플로우는 크게 세 단계로 구성됩니다:

1. **모델 준비**: 모델을 다운로드하고 ONNX로 내보냅니다.
2. **캘리브레이션 데이터셋 준비**: DOTA에서 대표성 있는 캘리브레이션 데이터를 구성합니다.
3. **모델 컴파일**: 캘리브레이션 데이터를 사용해 모델을 `.mxq`로 변환합니다.

## 단계 1: 모델 준비

`ultralytics`를 사용해 사전 학습된 OBB 모델을 다운로드하고 ONNX로 내보냅니다:

```bash
yolo export model=yolo11m-obb.pt format=onnx
```

내보내기가 완료되면 ONNX 모델은 현재 디렉토리에 `yolo11m-obb.onnx`로 저장됩니다.

## 단계 2: 캘리브레이션 데이터셋 준비

캘리브레이션 데이터셋은 모델이 실제로 받게 될 입력 분포를 반영해야 합니다. `yolo11m-obb.pt`는 [DOTA 데이터셋](https://captain-whu.github.io/DOTA/index.html) 기반의 항공 영상 분포에 맞춰 학습되었으므로, 이 튜토리얼에서는 DOTAv1 샘플을 캘리브레이션 데이터로 사용합니다.

Ultralytics에서 제공하는 아카이브를 바로 사용하여 데이터셋을 준비할 수 있습니다:

```bash
python prepare_dota.py
```

이 스크립트는 다음 작업을 수행합니다:

- `DOTAv1.zip`이 없으면 다운로드합니다.
- 아카이브를 `./DOTAv1`에 압축 해제합니다.
- 고정된 시드로 100장의 이미지를 무작위 선택합니다.
- 선택한 이미지를 `./dota-selected`로 복사합니다.

출력:

- `dota-selected`: 캘리브레이션 데이터셋 디렉토리

### 선택 가능한 데이터셋 인자

이미 데이터셋을 수동으로 다운로드한 경우에는 기존 파일을 재사용할 수 있습니다:

```bash
python prepare_dota.py --skip-download --zip-path ./DOTAv1.zip
```

이미 압축까지 풀어 두었고 캘리브레이션 서브셋만 만들고 싶다면 다음과 같이 실행합니다:

```bash
python prepare_dota.py --skip-download --extract-dir ./DOTAv1 --output-dir ./dota-selected --num-images 100
```

`--skip-download`를 사용하면 수동으로 압축을 푼 디렉토리처럼 스크립트의 `.extracted` 마커 파일이 없는
`--extract-dir`도 기존 추출본으로 그대로 재사용합니다.

## 단계 2-1 (선택): 이미지를 전처리된 텐서로 변환

캘리브레이션 데이터셋을 전처리된 `.npy` 텐서로 준비할 수도 있습니다. 모델에 맞는 전처리를 직접 적용한 텐서를 만들고 싶을 때 유용합니다.

`qbcompiler` v1.0.0 이후에는 일반적인 캘리브레이션 데이터셋 흐름만으로도 충분한 경우가 많습니다. 전처리를 명시적으로 제어해야 할 때만 이 단계를 사용하세요.

변환 스크립트는 다음과 같은 전처리 함수를 가정합니다:

- 입력으로 이미지 경로를 받음
- NumPy 텐서를 반환함
- 캘리브레이션용 `HWC` 형식 텐서를 생성함

예시 전처리 함수:

```python
def pre_ftn(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h0, w0 = img.shape[:2]
    r = min(1024 / h0, 1024 / w0)
    new_unpad = int(round(w0 * r)), int(round(h0 * r))
    dh, dw = 1024 - new_unpad[1], 1024 - new_unpad[0]

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

OBB 튜토리얼은 내보낸 `yolo11m-obb` 모델 입력에 맞추기 위해 `1024x1024` letterbox를 사용합니다.

스크립트는 `make_calib_man()`으로 텐서 데이터셋을 생성합니다:

```python
make_calib_man(
    pre_ftn=pre_ftn,
    data_dir=args.source_path,
    save_dir=os.path.dirname(args.npy_path),
    save_name=os.path.basename(args.npy_path),
    remove_npy=True,
)
```

스크립트 실행:

```bash
python convert_img_to_tensor.py
```

기본값으로 `./dota-selected`의 이미지를 읽고, 생성된 텐서 데이터셋은 `./calib_data_tensor`에 저장합니다.

## 단계 3: 모델 컴파일

컴파일 전에 필요한 전처리 단계를 확인합니다. Ultralytics에서 내보낸 OBB 모델은 letterbox 리사이즈를 사용하며, 이 튜토리얼은 캘리브레이션에서 같은 동작을 `1024x1024` letterbox로 맞춥니다.

Mobilint 컴파일 API는 캘리브레이션 중에 전처리 파이프라인을 적용합니다. 정규화 단계인 (`/255` 스케일링)은 `Uint8InputConfig`를 통해 MXQ 모델에 융합되므로, 런타임 모델은 `uint8` 입력을 직접 받을 수 있습니다. 반면 letterbox 같은 공간 변환은 융합되지 않으므로 런타임에서 계속 적용해야 합니다.

`model_compile.py`에서 전처리 파이프라인은 다음과 같이 정의됩니다:

```python
preprocess_pipeline = [
    {"op": "letterbox", "height": 1024, "width": 1024, "padValue": 114},
    {
        "op": "normalize",
        "scaleToUint8": True,
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "fuseIntoFirstLayer": True,
    },
]
preprocessing_config = PreprocessingConfig(
    apply=True,
    auto_convert_format=True,
    pipeline=preprocess_pipeline,
    input_configs={},
)
```

전처리 융합을 사용할 때는 MXQ 입력 타입을 `uint8`로 설정합니다:

```python
mxq_compile(
    # ... model, calibration data, backend, and target device settings
    preprocessing_config=preprocessing_config,
    uint8_input_config=Uint8InputConfig(apply=True, inputs=[]),
    calibration_config=calibration_config,
)
```

원래 입력 형식을 유지하고 싶다면 전처리 융합과 `Uint8InputConfig`를 모두 비활성화하면 됩니다.

이 예제는 다음 양자화 설정을 사용합니다:

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

설정을 구성한 후, 하드웨어에 맞는 `--target-device`를 지정해 `model_compile.py`를 실행합니다. 한 번 실행하면 양자화 MXQ 파일(`--save-path`)과 중간 MBLT 그래프(`--mblt-path`)가 함께 생성됩니다.

MXQ 컴파일 시 `model_compile.py`는 `torch.cuda.is_available()`이 참이면 CUDA를 사용하고, 그렇지 않으면 CPU로 자동 전환합니다. 따라서 GPU 지원 이미지와 CPU 전용 `qbcompiler` 이미지에서 모두 실행할 수 있으며, 선택된 호스트 디바이스는 컴파일 시작 전에 출력됩니다.

## 단계 3-1 (선택): 준비된 텐서 파일로 컴파일

이미 `.npy` 텐서 파일을 준비했다면, 원본 이미지와 전처리 파이프라인 대신 그 디렉토리를 `calib_data_path`로 사용할 수 있습니다.

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
- `--target-device`: 대상 NPU. 아래 표를 참고하세요. inference scheme 은 자동으로 선택됩니다 (`ARIES = all`, `REGULUS = single`).

출력:

- `--save-path`에 저장된 MXQ 모델
- `--mblt-path`에 저장된 MBLT 중간 그래프

### 대상 디바이스 선택 (`--target-device`)

| 사용자 | `--target-device` | 모델 |
| --- | --- | --- |
| ARIES | `aries-rb` | `yolo11m-obb` |
| REGULUS (2026-06 이후 고객) | `regulus-rb` | `yolo11m-obb` |

> **참고**: OBB 는 YOLO11 `yolo11m-obb` 모델을 사용하며, 이는 **구버전 REGULUS(`regulus-ra`, 2026-06 이전 고객)에서는 지원되지 않습니다** — 해당 세대는 YOLOv9 이하만 지원하고 OBB 는 그보다 상위 모델에서만 제공됩니다. `aries-rb` 또는 `regulus-rb` 를 사용하세요.

```bash
# ARIES
python model_compile.py --onnx-path ./yolo11m-obb.onnx --calib-data-path ./dota-selected --save-path ./yolo11m-obb.mxq --mblt-path ./yolo11m-obb.mblt --target-device aries-rb

# REGULUS (2026-06 이후 고객)
python model_compile.py --onnx-path ./yolo11m-obb.onnx --calib-data-path ./dota-selected --save-path ./yolo11m-obb.mxq --mblt-path ./yolo11m-obb.mblt --target-device regulus-rb
```

명령을 실행하면 현재 디렉토리에 MXQ(`yolo11m-obb.mxq`)와 MBLT(`yolo11m-obb.mblt`)가 저장됩니다.

## 이 튜토리얼의 파일

- `model_compile.py`: 선택한 `--target-device`에 맞춰 ONNX 모델을 MXQ / MBLT로 컴파일합니다
- `prepare_dota.py`: DOTAv1을 다운로드하거나 재사용하고 캘리브레이션 이미지를 준비합니다
- `convert_img_to_tensor.py`: DOTA 이미지를 캘리브레이션용 전처리 `.npy` 텐서로 변환합니다
- `README.md`: 이 예제의 영어 튜토리얼 문서입니다
- `README.KR.md`: 이 예제의 한국어 튜토리얼 문서입니다
