# 이미지 분류 모델 컴파일

본 튜토리얼은 Mobilint `qbcompiler`를 사용하여 이미지 분류 모델을 컴파일하는 상세 가이드를 제공합니다.

여기에서는 PyTorch에서 제공하는 ImageNet-1K 사전 학습 모델인 [ResNet-50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)을 사용합니다. 이 모델은 이미지를 1,000개의 클래스로 분류하는 표준적인 이미지 분류 모델입니다.

## 사전 요구사항

시작하기 전에 다음이 설치되어 있는지 확인해야 합니다:

- qbcompiler SDK 컴파일러 설치 (버전 >= 0.11 필요)
- ImageNet 데이터셋에 접근할 수 있는 HuggingFace 계정 (게이트된 데이터셋 사용 시)

## 개요

컴파일 과정은 크게 세 단계로 진행됩니다:

1. **모델 준비**: 모델을 다운로드하고 ONNX 형식으로 내보냅니다.
2. **캘리브레이션 데이터셋 준비**: ImageNet 데이터셋에서 대표 샘플을 추출하여 캘리브레이션 데이터를 생성합니다.
3. **모델 컴파일**: 캘리브레이션 데이터를 활용해 모델을 `.mxq` 형식으로 변환합니다.

또한 다음 Python 패키지를 설치해야 합니다:

```bash
pip install datasets
```

## Step 1: 모델 준비

먼저 모델을 준비해야 합니다. `torchvision` 라이브러리를 사용하여 사전 학습된 모델을 다운로드하고 `torch.onnx.export`를 통해 ONNX 형식으로 내보냅니다.

```python
import torch
from torchvision.models import resnet50, ResNet50_Weights

# 사전 학습된 가중치 사용:
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# 모델의 입력 형태에 맞춘 더미 입력 생성
input = torch.randn(1, 3, 224, 224)

# ONNX로 내보내기
torch.onnx.export(model, input, "resnet50.onnx")
```

위 코드(`prepare_model.py`)를 실행하면 내보낸 ONNX 모델이 현재 디렉토리에 `resnet50.onnx`로 저장됩니다.

## Step 2: 캘리브레이션 데이터셋 준비

캘리브레이션 데이터셋은 모델의 일반적인 입력 분포를 나타내는 이미지 집합입니다. 이 튜토리얼에서는 [ImageNet 데이터셋](https://huggingface.co/datasets/ILSVRC/imagenet-1k)을 사용합니다.

데이터셋을 사용하기 전에 [HuggingFace](https://huggingface.co/)에 계정을 등록하고 [데이터셋 페이지](https://huggingface.co/datasets/ILSVRC/imagenet-1k)에서 데이터셋 사용 동의를 수락하세요.

그런 다음 다음 명령을 사용하여 HuggingFace에 로그인하고 `<your_huggingface_token>`을 실제 HuggingFace 토큰으로 교체하세요:

```bash
hf auth login --token <your_huggingface_token>
```

HuggingFace 토큰을 모르는 경우 [HuggingFace 계정 설정](https://huggingface.co/settings/tokens)에서 찾을 수 있습니다.

그런 다음 HuggingFace에서 데이터셋을 다운로드하여 `imagenet-1k-selected` 디렉토리에 저장합니다. 이 스크립트는 데이터셋의 각 클래스에서 1000개의 이미지를 선택하여 `imagenet-1k-selected` 디렉토리에 저장합니다.

```bash
python prepare_imagenet.py
```

**작업 내용:**

- HuggingFace에서 데이터셋 다운로드
- 데이터셋의 각 클래스에서 1000개의 이미지 선택
- 선택한 이미지를 `imagenet-1k-selected` 디렉토리에 저장

**출력:**

- 선택한 이미지를 포함하는 `imagenet-1k-selected/` 디렉토리

선택된 이미지 데이터셋이 우리가 사용할 캘리브레이션 데이터셋입니다.

## Step 3: 모델 컴파일

모델을 컴파일하기 전, 해당 모델에 필요한 전처리 단계를 확인해야 합니다. [ResNet-50 공식 문서](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)에 따르면 모델의 입력 조건은 다음과 같습니다:
- 이미지의 짧은 쪽을 256픽셀로 리사이징 (Bilinear interpolation)
- 224x224 픽셀로 중앙 자르기 (Center crop)
- [0, 1] 범위로 픽셀 값 스케일링
- 평균 `[0.485, 0.456, 0.406]` 및 표준편차 `[0.229, 0.224, 0.225]`를 사용한 정규화

Mobilint 컴파일 API는 이러한 전처리를 내부적으로 수행하며, 정규화와 같은 작업을 MXQ 모델에 통합(fuse)하여 NPU 연산 효율을 극대화하도록 설계되었습니다.

`model_compile.py`에서 전처리 파이프라인을 다음과 같이 정의합니다. 이 파이프라인은 캘리브레이션에 사용되며 정규화 모듈을 딥러닝 모델에 병합합니다.

```python
preprocess_pipeline = [
    {"op": "resize", "height": 256, "width": 256, "mode": "bilinear"},
    {"op": "centerCrop", "height": 224, "width": 224},
    {
        "op": "normalize",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "scaleToUint8": True,  # [0, 255] -> [0, 1]
        "fuseIntoFirstLayer": True, # MXQ 내부에 병합
    },
]  # preprocessing operations for resnet 50

preprocessing_config = PreprocessingConfig(
    apply=True,
    auto_convert_format=True,
    pipeline=preprocess_pipeline,
    input_configs={},
)
```

또한 다음과 같은 전처리 설정 및 양자화 설정을 정의합니다.

```python
input_process_config = InputProcessConfig(
    uint8_input=Uint8InputConfig(apply=True, inputs=[]),
    image_channels=3,
    preprocessing=preprocessing_config,
)

quantization_config = QuantizationConfig.from_kwargs(
    quantization_method=1,  # 0 for per tensor, 1 for per channel
    quantization_output=0,  # 0 for layer, 1 for channel
    quantization_mode=2,  # maxpercentile
    percentile=0.9999,  # quantization percentile
    topk_ratio=0.01,  # quantization topk
)
```

설정을 구성한 후, 코드는 다음과 같이 실행할 수 있습니다.

```bash
python model_compile.py --onnx-path {path_to_onnx_model} --calib-data-path {path_to_calibration_dataset} --save-path {path_to_save_model}
```

**작업 내용:**

- ONNX 모델 로드
- 캘리브레이션 데이터 로드
- 모델을 `.mxq` 형식으로 컴파일

**매개변수:**

- `--onnx-path`: ONNX 모델 경로
- `--calib-data-path`: 캘리브레이션 데이터 경로
- `--save-path`: MXQ 모델을 저장할 경로

**출력:**

- 컴파일된 모델을 포함하는 `{path_to_save_model}` 파일 경로

예를 들어, 명령은 다음과 같습니다:

```bash
python model_compile.py --onnx-path ./resnet50.onnx --calib-data-path ./imagenet-1k-selected --save-path ./resnet50.mxq 
```

위 명령을 실행하면 컴파일된 모델이 현재 디렉토리에 `resnet50.mxq`로 저장됩니다.
