# 이미지 분류 모델 컴파일

본 튜토리얼은 모빌린트 `qbcompiler`를 사용하여 이미지 분류 모델을 컴파일하는 방법에 대한 포괄적인 가이드를 제공합니다.

`torchvision`을 통해 제공되는 [ResNet-50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) 모델을 사용할 것입니다. ImageNet-1K 데이터셋으로 사전 학습된 이 모델은 이미지를 1,000개의 고유한 카테고리로 분류하기 위한 표준 벤치마크 모델입니다.

## 사전 준비

시작하기 전에 다음이 설치되어 있는지 확인하세요:

- qbcompiler
- ImageNet 데이터셋에 접근 가능한 HuggingFace 계정 (gated 데이터셋 사용을 위해)

## 개요

컴파일 워크플로우는 세 가지 주요 단계로 구성됩니다:

1. **모델 준비**: 모델을 다운로드하고 ONNX 형식으로 내보냅니다.
2. **캘리브레이션 데이터셋 준비**: ImageNet에서 대표적인 캘리브레이션 데이터셋을 생성합니다.
3. **모델 컴파일**: 캘리브레이션 데이터를 사용하여 모델을 `.mxq` 형식으로 변환합니다.

또한, 다음 패키지들을 설치해야 합니다:

```bash
pip install datasets
```

## 단계 1: 모델 준비

먼저 모델을 준비해야 합니다. `torchvision` 라이브러리를 사용하여 사전 학습된 모델을 다운로드하고, `torch.onnx.export`를 통해 ONNX 형식으로 내보냅니다.

```python
import torch
from torchvision.models import resnet50, ResNet50_Weights

# 사전 학습된 가중치 사용:
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# 모델의 입력 형태를 기반으로 더미 입력(dummy input) 생성
input = torch.randn(1, 3, 224, 224)

# ONNX로 내보내기
torch.onnx.export(model, input, "resnet50.onnx")
```

위의 코드(`prepare_model.py`)를 실행하면, 내보낸 ONNX 모델이 현재 디렉토리에 `resnet50.onnx`로 저장됩니다.

## 단계 2: 캘리브레이션 데이터셋 준비

캘리브레이션 데이터셋은 모델의 전형적인 입력 분포를 나타내는 이미지들의 집합입니다. 본 튜토리얼에서는 [ImageNet 데이터셋](https://huggingface.co/datasets/ILSVRC/imagenet-1k)을 사용할 것입니다.

데이터셋을 사용하기 전에 [HuggingFace](https://huggingface.co/)에 가입하고, [데이터셋 페이지](https://huggingface.co/datasets/ILSVRC/imagenet-1k)에서 데이터셋 사용 약관에 동의해야 합니다.

그 다음, 아래 명령어를 사용하여 HuggingFace에 로그인하고 `<your_huggingface_token>`을 실제 HuggingFace 토큰으로 변경하세요:

```bash
hf auth login --token <your_huggingface_token>
```

HuggingFace 토큰을 모르는 경우, [HuggingFace 계정 설정](https://huggingface.co/settings/tokens)에서 확인할 수 있습니다.

그런 다음 HuggingFace에서 데이터셋을 다운로드하고 `imagenet-1k-selected` 디렉토리에 저장합니다. 이 스크립트는 데이터셋의 각 클래스에서 1장의 이미지를 선택하여 총 1,000개의 이미지 파일을 `imagenet-1k-selected` 디렉토리에 저장합니다.

```bash
python prepare_imagenet.py
```

**수행 작업:**

- HuggingFace에서 데이터셋 다운로드
- 데이터셋의 각 클래스에서 1장의 이미지를 선택
- 선택한 이미지를 `imagenet-1k-selected` 디렉토리에 저장

**출력:**

- 선택한 이미지가 포함된 `imagenet-1k-selected/` 디렉토리

선택한 이미지 데이터셋이 우리가 사용할 캘리브레이션 데이터셋입니다.

## 단계 3: 모델 컴파일

컴파일을 실행하기 전에 모델에 필요한 전처리 과정을 확인하세요. [공식 ResNet-50 문서](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)에 따르면 필요한 전처리 작업은 다음과 같습니다:

- 짧은 쪽을 256 픽셀로 크기 조정 (Bilinear 보간법 사용)
- 224x224 픽셀로 중앙 자르기(Center Crop)
- `[0, 1]` 범위로 스케일링 설정
- 평균 `[0.485, 0.456, 0.406]` 및 표준 편차 `[0.229, 0.224, 0.225]`로 정규화(Normalization)

모빌린트 컴파일 API는 전처리 파이프라인을 캘리브레이션 과정에서 사용합니다. 정규화 연산(mean/std 및 /255 스케일링)은 `fuseIntoFirstLayer`와 `Uint8InputConfig`를 통해 MXQ 모델에 융합되며, 런타임에서 uint8 입력을 직접 받을 수 있게 합니다. 공간 변환(resize, centerCrop)은 융합되지 않으므로 런타임에서 직접 수행해야 합니다.

`model_compile.py`에서 다음과 같이 전처리 파이프라인을 정의합니다. 이 파이프라인은 캘리브레이션 이미지에 적용됩니다.

```python
preprocess_pipeline = [
    {"op": "resize", "height": 256, "width": 256, "mode": "bilinear"},
    {"op": "centerCrop", "height": 224, "width": 224},
    {
        "op": "normalize",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "scaleToUint8": True,  # [0, 255] -> [0, 1]
        "fuseIntoFirstLayer": True, # MXQ 모듈로 융합
    },
]  # resnet 50을 위한 전처리 작업들

preprocessing_config = PreprocessingConfig(
    apply=True,
    auto_convert_format=True,
    pipeline=preprocess_pipeline,
    input_configs={},
)
```

또한 모델을 양자화하는 데 사용되는 양자화 구성을 다음과 같이 정의합니다.

```python
calibration_config = CalibrationConfig(
        method=1,  # 0: 텐서, 1: 채널
        output=0,  # 0: 레이어, 1: 채널
        mode=1,  # maxpercentile
        max_percentile={
            "percentile": 0.9999,  # quantization percentile
            "topk_ratio": 0.01,  # quantization topk
        },
    )
```

설정을 구성한 후 대상 디바이스에 맞는 스크립트를 실행합니다.

**파라미터:**

- `--onnx-path`: ONNX 모델 경로
- `--calib-data-path`: 캘리브레이션 데이터 경로
- `--save-path`: MXQ 모델을 저장할 경로

**출력:**

- 컴파일된 모델이 포함된 `{path_to_save_model}` 파일 경로

### ARIES

ARIES는 `inference_scheme="all"`을 사용하여 하나의 MXQ 모델에서 여러 추론 스킴을 지원합니다.

```bash
python model_compile.py --onnx-path ./resnet50.onnx --calib-data-path ./imagenet-1k-selected --save-path ./resnet50.mxq
```

위의 명령어를 실행하면, 컴파일된 모델이 현재 디렉토리에 `resnet50.mxq`로 저장됩니다.

### REGULUS

REGULUS는 `inference_scheme="single"`만 지원합니다. `model_compile_regulus.py`를 사용하세요.

```bash
python model_compile_regulus.py --onnx-path ./resnet50.onnx --calib-data-path ./imagenet-1k-selected --save-path ./resnet50.mxq
```

위의 명령어를 실행하면, 컴파일된 모델이 현재 디렉토리에 `resnet50.mxq`로 저장됩니다.
