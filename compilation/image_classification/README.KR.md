# 이미지 분류 모델 컴파일

이 튜토리얼은 Mobilint `qbcompiler`를 사용해 이미지 분류 모델을 컴파일하는 방법을 설명합니다.

예제로는 `torchvision`의 [ResNet-50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)을 사용합니다. 이 모델은 ImageNet-1K로 사전 학습된 대표적인 이미지 분류 모델이며, 1,000개 카테고리 분류의 표준 벤치마크로 널리 사용됩니다.

## 사전 준비

시작하기 전에 다음 항목이 준비되어 있어야 합니다.

- `qbcompiler`
- gated ImageNet 데이터셋에 접근할 수 있는 Hugging Face 계정

또한 데이터셋 다운로드에 필요한 패키지를 설치하세요.

```bash
pip install datasets
```

## 개요

전체 워크플로우는 다음 세 단계로 구성됩니다.

1. **모델 준비**: ResNet-50 모델을 다운로드하고 ONNX로 내보냅니다.
2. **캘리브레이션 데이터셋 준비**: ImageNet에서 대표성 있는 캘리브레이션 데이터를 구성합니다.
3. **모델 컴파일**: 캘리브레이션 데이터를 사용해 ONNX 모델을 `.mxq`로 변환합니다.

## 단계 1: 모델 준비

`torchvision`으로 사전 학습된 모델을 다운로드한 뒤, `torch.onnx.export`를 사용해 ONNX 형식으로 내보냅니다.

```python
import torch
from torchvision.models import ResNet50_Weights, resnet50

# 사전 학습된 가중치 사용
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# 모델 입력 크기에 맞는 더미 입력 생성
dummy_input = torch.randn(1, 3, 224, 224)

# ONNX로 내보내기
torch.onnx.export(model, (dummy_input,), "resnet50.onnx")
```

스크립트 실행:

```bash
python prepare_model.py
```

실행 후 현재 디렉토리에 `resnet50.onnx`가 생성됩니다.

## 단계 2: 캘리브레이션 데이터셋 원본 준비

캘리브레이션 데이터셋은 양자화를 위한 활성값 통계를 수집하는 데 사용됩니다. 이 ResNet-50 예제에서는 [ImageNet 데이터셋](https://huggingface.co/datasets/ILSVRC/imagenet-1k)을 사용합니다.

데이터셋을 다운로드하기 전에 다음을 완료하세요.

- [Hugging Face](https://huggingface.co/) 계정 생성
- [ImageNet-1K 데이터셋 페이지](https://huggingface.co/datasets/ILSVRC/imagenet-1k)에서 사용 약관 동의

그다음 Hugging Face 토큰으로 로그인합니다.

```bash
hf auth login --token <your_huggingface_token>
```

토큰을 모르는 경우 [Hugging Face 토큰 설정 페이지](https://huggingface.co/settings/tokens)에서 확인할 수 있습니다.

이후 데이터셋 준비 스크립트를 실행합니다.

```bash
python prepare_imagenet.py
```

이 스크립트는 다음 작업을 수행합니다.

- Hugging Face에서 validation split 다운로드
- 1,000개 클래스 각각에서 이미지 1장 선택
- 선택한 이미지를 `imagenet-1k-selected/`에 저장

출력:

- `imagenet-1k-selected/` 디렉토리

이 디렉토리가 다음 단계에서 사용할 캘리브레이션 이미지 데이터셋입니다.

## 단계 2-1 (선택): 이미지를 전처리된 텐서로 변환

캘리브레이션 데이터는 전처리된 `.npy` 텐서 형태로도 준비할 수 있습니다. 모델이 사용자 정의 전처리를 요구하고, 직접 텐서 입력을 생성하고 싶을 때 유용합니다.

`qbcompiler` v1.0.0부터는 표준화된 캘리브레이션 데이터 생성 흐름이 제공되므로, 일반적으로 이 단계는 생략할 수 있습니다. 전처리를 직접 제어해야 하는 경우에만 사용하세요.

이 변환 스크립트는 다음 조건의 전처리 함수를 가정합니다.

- 입력은 이미지 경로
- 출력은 NumPy 텐서
- 캘리브레이션용 텐서 형식은 `HWC`

전처리 함수 예시는 다음과 같습니다.

```python
def pre_ftn(img_path):
    img = Image.open(img_path).convert("RGB")
    preprocess_pipeline = [
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop((224, 224)),
        T.ToTensor(),  # [0, 255] -> [0, 1]
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    preprocess = T.Compose(preprocess_pipeline)
    tensor = cast(torch.Tensor, preprocess(img))
    return tensor.permute((1, 2, 0)).numpy()  # (C, H, W) -> (H, W, C)
```

스크립트는 `make_calib_man()`을 사용해 텐서 데이터셋을 생성합니다.

```python
make_calib_man(
    pre_ftn=pre_ftn,
    data_dir=args.source_path,
    save_dir=os.path.dirname(args.npy_path),
    save_name=os.path.basename(args.npy_path),
    remove_npy=True,  # 새 .npy 파일 저장 전에 기존 결과 삭제
)
```

스크립트 실행:

```bash
python convert_img_to_tensor.py
```

기본값 기준으로 입력 이미지는 `./imagenet-1k-selected`에서 읽고, 결과 텐서는 `./calib_data_tensor` 아래에 저장됩니다.

## 단계 3: 모델 컴파일

컴파일 전에 모델이 요구하는 전처리를 먼저 확인해야 합니다. [공식 ResNet-50 문서](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)에 따르면 필요한 전처리는 다음과 같습니다.

- 짧은 변을 256픽셀로 bilinear resize
- `224x224` center crop
- 픽셀 값을 `[0, 1]` 범위로 스케일링
- 평균 `[0.485, 0.456, 0.406]`으로 정규화
- 표준편차 `[0.229, 0.224, 0.225]`로 정규화

이 튜토리얼에서는 `qbcompiler`의 표준 전처리 파이프라인으로 해당 전처리를 적용합니다.

```python
preprocess_pipeline = [
    {"op": "resize", "height": 256, "width": 256, "mode": "bilinear"},
    {"op": "centerCrop", "height": 224, "width": 224},
    {
        "op": "normalize",
        "scaleToUint8": True,  # [0, 255] -> [0, 1]
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "fuseIntoFirstLayer": True,
    },
]  # ResNet-50용 전처리 연산

preprocessing_config = PreprocessingConfig(
    apply=True,
    auto_convert_format=True,
    pipeline=preprocess_pipeline,
    input_configs={},
)
```

정규화 단계와 `/255` 스케일링은 `fuseIntoFirstLayer`와 `Uint8InputConfig`를 통해 MXQ 모델에 융합됩니다. 따라서 런타임에서 컴파일된 모델은 `uint8` 입력을 직접 받을 수 있습니다. 반면 `resize`와 `centerCrop` 같은 공간 변환은 융합되지 않으므로 런타임에서 별도로 적용해야 합니다.

전처리 융합을 사용할 때는 MXQ 입력 타입도 `uint8`로 설정해야 합니다.

```python
# ONNX -> MXQ: NPU에서 실행되는 양자화 패키지
mxq_compile(
    # ... model, calibration data, backend, and target device settings
    preprocessing_config=preprocessing_config,
    uint8_input_config=Uint8InputConfig(apply=True, inputs=[]),
    calibration_config=calibration_config,
)
```

원래 입력 형식을 유지하고 싶다면 `fuseIntoFirstLayer`와 `Uint8InputConfig`를 모두 비활성화하면 됩니다.

예제에서는 다음과 같은 양자화 설정도 사용합니다.

```python
calibration_config = CalibrationConfig(
    method=1,  # 0 for per tensor, 1 for per channel
    output=0,  # 0 for layer, 1 for channel
    mode=1,  # maxpercentile
    max_percentile={
        "percentile": 0.9999,  # quantization percentile
        "topk_ratio": 0.01,  # quantization top-k ratio
    },
)
```

설정이 끝나면 `--target-device`를 지정해 `model_compile.py`를 실행합니다. 같은 스크립트로 ARIES와 REGULUS를 모두 지원합니다.

## 단계 3-1 (선택): 준비된 텐서 파일로 컴파일

이미 전처리된 `.npy` 텐서 파일이 준비되어 있다면, 원본 이미지와 전처리 파이프라인 대신 해당 디렉토리를 `calib_data_path`로 지정해 사용할 수 있습니다.

```python
mxq_compile(
    model=args.onnx_path,
    calib_data_path=args.calib_data_path,  # .npy 파일 디렉토리 또는 파일 목록이 적힌 .txt
    save_path=args.save_path,
    image_channels=3,  # 필요 시 grayscale calibration image를 RGB로 변환
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
- `--target-device` (필수): 대상 NPU. 아래 표를 참고하세요. inference scheme은 자동으로 선택됩니다 (`ARIES = all`, `REGULUS = single`).

출력:

- `--save-path`에 저장되는 MXQ 모델 (`onnx -> mxq`, 양자화된 NPU 패키지)
- `--mblt-path`에 저장되는 MBLT 그래프 (`onnx -> mblt`, 양자화 전 중간 그래프)

### 대상 디바이스 선택 (`--target-device`)

| 사용자 | `--target-device` |
| --- | --- |
| ARIES | `aries-rb` |
| REGULUS (2026-06 이전 고객) | `regulus-ra` |
| REGULUS (2026-06 이후 고객) | `regulus-rb` |

```bash
# ARIES
python model_compile.py --onnx-path ./resnet50.onnx --calib-data-path ./imagenet-1k-selected --save-path ./resnet50.mxq --mblt-path ./resnet50.mblt --target-device aries-rb

# REGULUS (2026-06 이전 고객)
python model_compile.py --onnx-path ./resnet50.onnx --calib-data-path ./imagenet-1k-selected --save-path ./resnet50.mxq --mblt-path ./resnet50.mblt --target-device regulus-ra

# REGULUS (2026-06 이후 고객)
python model_compile.py --onnx-path ./resnet50.onnx --calib-data-path ./imagenet-1k-selected --save-path ./resnet50.mxq --mblt-path ./resnet50.mblt --target-device regulus-rb
```

명령이 완료되면 현재 디렉토리에 `resnet50.mxq`와 `resnet50.mblt`가 저장됩니다.
