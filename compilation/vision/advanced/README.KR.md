# 이미지 분류 모델을 이용한 심화 테스트 (Advanced Test with Image Classification Model)

이 튜토리얼은 모빌린트(Mobilint) Qubee 컴파일러를 사용하여 이미지 분류 모델을 컴파일하는 방법에 대한 자세한 지침을 제공합니다.

이 튜토리얼에서는 PyTorch에서 제공하는 ImageNet 데이터셋으로 사전 학습된 [ResNet-50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) 모델을 사용합니다. 이 모델은 이미지를 1000개의 클래스로 분류할 수 있는 간단한 이미지 분류 모델입니다.

## 사전 요구 사항 (Prerequisites)

시작하기 전에 다음 항목이 설치되어 있는지 확인하십시오.

- Qubee SDK 컴파일러 (버전 0.11 이상 필요)
- ImageNet 데이터셋에 액세스할 수 있는 HuggingFace 계정 (gated 데이터셋을 사용하는 경우)

## 개요 (Overview)

컴파일 프로세스는 다음 세 가지 주요 단계로 구성됩니다.

1. **모델 준비 (Model Preparation)**: 모델 다운로드 및 ONNX 형식으로 내보내기
2. **보정 데이터셋 생성 (Calibration Dataset Generation)**: ImageNet 데이터셋에서 보정 데이터 생성
3. **모델 컴파일 (Model Compilation)**: 보정 데이터를 사용하여 모델을 `.mxq` 형식으로 변환

또한 다음 패키지를 설치해야 합니다.

```bash
pip install datasets
```

## 1단계: 모델 준비 (Step 1: Model Preparation)

먼저 모델을 준비해야 합니다. `torchvision` 라이브러리를 사용하여 사전 학습된 모델을 다운로드하고 `torch.onnx.export`를 통해 ONNX 형식으로 내보냅니다.

```python
import torch
from torchvision.models import resnet50, ResNet50_Weights

# Using pretrained weights:
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()
# make dummy input depending on the model's input shape
input = torch.randn(1, 3, 224, 224)
# export to onnx
torch.onnx.export(model, input, "resnet50.onnx")
```

위의 코드(`prepare_model.py`)를 실행하면 내보내진 ONNX 모델이 현재 디렉토리에 `resnet50.onnx`로 저장됩니다.

## 2단계: 보정 데이터셋 준비 (Step 2: Calibration Dataset Preparation)

보정 데이터셋을 준비하려면 다음 명령어를 실행하여 원본 소스 이미지 세트를 가져옵니다.

```bash
python prepare_imagenet.py
```

이 코드는 5개의 폴더를 생성합니다.

- `imagenet-1k-100cls-100`
- `imagenet-1k-10cls-10`
- `imagenet-1k-20cls-100`
- `imagenet-1k-5cls-100`
- `imagenet-1k-1000cls-1000`

각 폴더는 특정 클래스에 대한 이미지들을 포함합니다. 각 데이터셋에서 다음 명령어를 실행하여 보정 데이터를 생성할 수 있습니다.

```bash
python prepare_calib.py --data_dir {path_to_imagenet_dataset}
```

## 3단계: 모델 컴파일 (Step 3: Model Compilation)

보정 데이터셋과 모델이 준비되면 모델을 컴파일할 수 있습니다.

```bash
python model_compile.py --onnx_path {path_to_onnx_model} --calib_data_path {path_to_calibration_dataset} --save_path {path_to_save_model} --quant_percentile {quantization_percentile} --topk_ratio {topk_ratio} --inference_scheme {inference_scheme}
```

**기능 설명:**

- ONNX 모델 로드
- 보정 데이터 로드
- 모델을 `.mxq` 형식으로 컴파일

**매개변수:**

- `--onnx_path`: ONNX 모델 경로
- `--calib_data_path`: 보정 데이터 경로
- `--save_path`: MXQ 모델 저장 경로
- `--quant_percentile`: 양자화 백분위수 (Quantization percentile)
- `--topk_ratio`: Top-k 비율 (Top-k ratio)
- `--inference_scheme`: 추론 스킴 (inference scheme) (single, multi, global, global4, global8)

**출력:**

- 컴파일된 모델이 포함된 `{path_to_save_model}` 파일 경로

예를 들어, 명령어는 다음과 같습니다.

```bash
python model_compile.py --onnx_path ./resnet50.onnx --calib_data_path ./resnet50_imagenet-1k-5cls-100 --save_path ./resnet50_5cls_100_9999_01.mxq --quant_percentile 0.9999 --topk_ratio 0.01 --inference_scheme single
```

위의 명령어를 실행하면 컴파일된 모델이 현재 디렉토리에 `resnet50_5cls_100_9999_01.mxq`로 저장됩니다.
