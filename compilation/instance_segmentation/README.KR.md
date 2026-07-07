# 인스턴스 분할 모델 컴파일

이 튜토리얼은 Mobilint `qbcompiler`로 인스턴스 분할 모델을 컴파일하는 방법을 설명합니다.

예제에서는 Ultralytics의 COCO 사전 학습 모델인 [YOLO11m-seg](https://docs.ultralytics.com/models/yolo11/)를 사용합니다. 이 모델은 개별 객체를 검출하고 각 객체의 마스크를 예측하는 인스턴스 분할 모델입니다.

## 사전 준비

시작하기 전에 다음 항목을 준비하세요:

- `qbcompiler` v1.0.0
- gated COCO 데이터셋에 접근할 수 있는 Hugging Face 계정

필요한 Python 패키지는 다음과 같이 설치합니다:

```bash
pip install ultralytics aiohttp aiofiles
```text

## 개요

전체 워크플로우는 세 단계로 구성됩니다:

1. **모델 준비**: 사전 학습 모델을 다운로드하고 ONNX로 export 합니다.
2. **캘리브레이션 데이터 준비**: COCO에서 대표성 있는 캘리브레이션 데이터를 만듭니다.
3. **모델 컴파일**: 캘리브레이션 데이터를 사용해 ONNX 모델을 `.mxq`로 변환합니다.

## 단계 1: 모델 준비

`ultralytics` CLI를 사용해 사전 학습 모델을 ONNX로 export 합니다:

```bash
yolo export model=yolo11m-seg.pt format=onnx
```text

명령이 끝나면 현재 디렉토리에 `yolo11m-seg.onnx`가 생성됩니다.

## 단계 2: 캘리브레이션 데이터셋 준비

캘리브레이션 데이터셋은 모델이 실제로 받게 될 입력 분포를 최대한 잘 대표해야 합니다. YOLO11m-seg는 [COCO 데이터셋](https://cocodataset.org/#download)으로 학습되었기 때문에, 이 튜토리얼에서도 COCO 샘플을 사용합니다.

데이터를 다운로드하기 전에, 데이터셋 접근 권한이 있는 Hugging Face 토큰으로 로그인하세요:

```bash
hf auth login --token <your_huggingface_token>
```text

토큰이 없다면 [Hugging Face 토큰 설정 페이지](https://huggingface.co/settings/tokens)에서 생성하거나 확인할 수 있습니다.

`prepare_coco.py`를 사용하면 COCO 이미지 일부를 무작위로 선택해 `coco-selected` 디렉토리에 다운로드할 수 있습니다:

```bash
python prepare_coco.py
```text

스크립트 동작:

- Hugging Face에서 COCO 이미지 URL을 가져옵니다
- 캘리브레이션용 이미지를 무작위로 선택합니다
- 선택한 이미지를 `coco-selected`에 저장합니다

**출력:**

- `coco-selected`: 캘리브레이션 이미지 디렉토리

이 디렉토리가 다음 단계에서 사용할 캘리브레이션 데이터셋입니다.

## 단계 2-1 (선택): 이미지를 전처리된 텐서로 변환

캘리브레이션 입력을 전처리된 `.npy` 텐서 형태로 직접 준비할 수도 있습니다. 모델에 맞는 커스텀 전처리가 필요하거나, 텐서 생성 과정을 직접 제어하고 싶을 때 유용합니다.

`qbcompiler` v1.0.0부터는 표준화된 캘리브레이션 데이터 생성 흐름이 제공되므로, 대부분의 경우 이 단계는 건너뛰어도 됩니다. 전처리를 수동으로 제어해야 할 때만 사용하세요.

변환 스크립트는 다음 조건을 만족하는 전처리 함수를 가정합니다:

- 입력은 이미지 경로
- 출력은 NumPy 텐서
- 캘리브레이션용 텐서는 `HWC` 형식

전처리 함수 예시는 다음과 같습니다:

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
```text

스크립트는 `make_calib_man()`을 호출해 텐서 데이터셋을 생성합니다:

```python
make_calib_man(
    pre_ftn=pre_ftn,
    data_dir=args.source_path,
    save_dir=os.path.dirname(args.npy_path),
    save_name=os.path.basename(args.npy_path),
    remove_npy=True,  # Clean the destination before writing new .npy files.
)
```text

변환 스크립트 실행:

```bash
python convert_img_to_tensor.py
```text

기본값 기준으로 `./coco-selected`에서 이미지를 읽고, 생성된 텐서는 `./calib_data_tensor` 아래에 저장됩니다.

## 단계 3: 모델 컴파일

컴파일 전에 필요한 전처리 과정을 먼저 확인하세요. YOLO 계열 모델은 일반적으로 [Ultralytics 저장소](https://github.com/ultralytics/ultralytics)에서 설명하는 `LetterBox` 연산을 사용합니다.

Mobilint 컴파일 API는 이 전처리를 내부적으로 수행하며, NPU 효율을 높이기 위해 해당 연산을 MXQ 모델에 fuse할 수 있습니다.

`model_compile.py`에서는 다음과 같이 전처리 파이프라인을 정의합니다:

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
```text

정규화 과정의 일부로 `letterbox` 단계에는 `1/255` 스케일링도 포함됩니다. 이 전처리는 `Uint8InputConfig`를 사용해 MXQ 모델 안으로 fuse할 수 있습니다.

전처리 fuse를 사용할 때는 MXQ 입력 타입을 `uint8`로 설정하세요:

```python
# ONNX -> MXQ: quantized package that runs on the NPU
mxq_compile(
    # ... model, calibration data, backend, and target device settings
    preprocessing_config=preprocessing_config,
    uint8_input_config=Uint8InputConfig(apply=True, inputs=[]),
    calibration_config=calibration_config,
)
```text

원래 입력 형식을 유지하고 싶다면 전처리 fuse와 `Uint8InputConfig`를 모두 비활성화하면 됩니다.

예제에서는 다음과 같은 양자화 설정도 사용합니다:

```python
calibration_config = CalibrationConfig(
    method=1,  # 0 for per tensor, 1 for per channel
    output=1,  # 0 for layer, 1 for channel
    mode=1,  # maxpercentile
    max_percentile={
        "percentile": 0.9999,  # quantization percentile
        "topk_ratio": 0.01,  # quantization topk
    },
)
```text

설정을 마친 뒤 하드웨어에 맞는 `--target-device`를 지정해 `model_compile.py`를 실행합니다. 스크립트 한 번으로 양자화된 MXQ 파일(`--save-path`)과 중간 MBLT 그래프(`--mblt-path`)를 모두 생성합니다.

## 단계 3-1 (선택): 준비된 텐서 파일로 컴파일

이미 `.npy` 텐서를 만들어 두었다면, 원본 이미지와 전처리 파이프라인 대신 해당 디렉토리를 `calib_data_path`로 사용할 수 있습니다.

```python
mxq_compile(
    model=args.onnx_path,
    calib_data_path=args.calib_data_path,  # Directory of .npy files, or a .txt file listing them
    save_path=args.save_path,
    image_channels=3,  # Convert grayscale calibration images to RGB if needed
    backend="onnx",
    device="gpu",
    target_device=args.target_device,
    inference_scheme=inferece_sheme,
    calibration_config=calibration_config,
)
```text

파라미터:

- `--onnx-path`: ONNX 모델 경로
- `--calib-data-path`: 캘리브레이션 데이터 경로
- `--save-path`: MXQ 모델 저장 경로 (`onnx -> mxq`)
- `--mblt-path`: MBLT 중간 그래프 저장 경로 (`onnx -> mblt`)
- `--target-device` (필수): 대상 NPU. 아래 표를 참고하세요. inference scheme은 자동으로 선택됩니다 (`ARIES = all`, `REGULUS = single`).

출력:

- `--save-path`에 저장되는 MXQ 모델 (`onnx -> mxq`, 양자화된 NPU 패키지)
- `--mblt-path`에 저장되는 MBLT 중간 그래프 (`onnx -> mblt`, 양자화 전 그래프)

### 대상 디바이스 선택 (`--target-device`)

사용해야 하는 모델은 대상 디바이스에 따라 다릅니다:

- 구형 REGULUS 하드웨어(`regulus-ra`, 2026-06 이전 고객)는 YOLOv9 이하만 지원하므로 YOLOv8 segmentation 모델을 사용합니다.
- ARIES(`aries-rb`)와 신형 REGULUS 하드웨어(`regulus-rb`, 2026-06 이후 고객)는 YOLO11 segmentation 모델을 사용합니다.

| 사용자 | `--target-device` | 모델 |
|---|---|---|
| ARIES | `aries-rb` | `yolo11m-seg` |
| REGULUS (2026-06 이전 고객) | `regulus-ra` | `yolov8m-seg` |
| REGULUS (2026-06 이후 고객) | `regulus-rb` | `yolo11m-seg` |

먼저 디바이스에 맞는 모델을 export 하세요. 1단계는 `yolo11m-seg` 기준이며, `regulus-ra`를 사용할 때는 `yolov8m-seg`를 export 해야 합니다:

```bash
# YOLO11 seg (for aries-rb / regulus-rb)
yolo export model=yolo11m-seg.pt format=onnx

# YOLOv8 seg (for regulus-ra)
yolo export model=yolov8m-seg.pt format=onnx
```text

```bash
# ARIES
python model_compile.py --onnx-path ./yolo11m-seg.onnx --calib-data-path ./coco-selected --save-path ./yolo11m-seg.mxq --mblt-path ./yolo11m-seg.mblt --target-device aries-rb

# REGULUS (2026-06 이전 고객)
python model_compile.py --onnx-path ./yolov8m-seg.onnx --calib-data-path ./coco-selected --save-path ./yolov8m-seg.mxq --mblt-path ./yolov8m-seg.mblt --target-device regulus-ra

# REGULUS (2026-06 이후 고객)
python model_compile.py --onnx-path ./yolo11m-seg.onnx --calib-data-path ./coco-selected --save-path ./yolo11m-seg.mxq --mblt-path ./yolo11m-seg.mblt --target-device regulus-rb
```text

명령이 끝나면 현재 디렉토리에 해당 MXQ와 MBLT 파일이 저장됩니다.
