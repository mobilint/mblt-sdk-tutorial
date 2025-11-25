# 이미지 분류 모델 컴파일

이 튜토리얼은 Mobilint qubee 컴파일러를 사용하여 이미지 분류 모델을 컴파일하는 방법에 대한 자세한 지침을 제공합니다.

이 튜토리얼에서는 PyTorch에서 개발한 ImageNet 데이터셋으로 사전 학습된 [ResNet-50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) 모델을 사용합니다. 이 모델은 이미지를 1000개의 클래스로 분류할 수 있는 간단한 이미지 분류 모델입니다.

## 사전 요구사항

시작하기 전에 다음이 설치되어 있는지 확인해야 합니다:

- qubee SDK 컴파일러 설치 (버전 >= 0.11 필요)
- ImageNet 데이터셋에 접근할 수 있는 HuggingFace 계정 (게이트된 데이터셋 사용 시)

## 개요

컴파일 프로세스는 세 가지 주요 단계로 구성됩니다:

1. **모델 준비**: 모델을 다운로드하고 ONNX 형식으로 내보내기
2. **캘리브레이션 데이터셋 생성**: ImageNet 데이터셋에서 캘리브레이션 데이터 생성
3. **모델 컴파일**: 캘리브레이션 데이터를 사용하여 모델을 `.mxq` 형식으로 변환

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
# 모델의 입력 형태에 따라 더미 입력 생성
input = torch.randn(1, 3, 224, 224)
# onnx로 내보내기
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

선택한 이미지로 캘리브레이션 데이터셋을 생성할 수 있습니다. 캘리브레이션 데이터셋을 생성하기 전에 원본 모델이 사용하는 전처리를 확인해야 합니다. 전처리 정보는 원본 [ResNet-50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) 페이지에서 찾을 수 있습니다. 모델이 사용하는 전처리 작업은 다음과 같습니다: 양선형 보간을 사용하여 이미지의 짧은 쪽을 256픽셀로 크기 조정, 224x224픽셀로 중앙 자르기, 이미지를 [0, 1] 범위로 재조정, 평균 [0.485, 0.456, 0.406] 및 표준 편차 [0.229, 0.224, 0.225]로 이미지 정규화.

qubee 컴파일러의 유틸리티 함수 `make_calib`는 이러한 종류의 표준 전처리 작업으로 캘리브레이션 데이터셋을 생성하도록 설계되었습니다. `prepare_calib.py` 스크립트는 전처리 구성 파일 `resnet50.yaml`과 `imagenet-1k-selected` 디렉토리의 원본 캘리브레이션 데이터를 읽어 위에서 정의한 전처리 작업으로 캘리브레이션 데이터셋을 생성하기 위해 이 함수를 사용합니다.

```bash
python prepare_calib.py
```

**작업 내용:**

- 전처리 구성 파일 `resnet50.yaml` 읽기
- `imagenet-1k-selected` 디렉토리에서 원본 캘리브레이션 데이터 읽기
- 위에서 정의한 전처리 작업으로 캘리브레이션 데이터셋 생성
- 캘리브레이션 데이터셋을 `resnet50_cali` 디렉토리에 저장

**출력:**

- 캘리브레이션 데이터셋을 포함하는 `resnet50_cali/` 디렉토리
- 캘리브레이션 데이터셋의 경로를 포함하는 `resnet50_cali.txt` 파일

캘리브레이션 데이터셋은 `resnet50_cali` 디렉토리에 저장됩니다. `resnet50_cali.txt` 파일에 기록된 경로도 확인할 수 있습니다.

## Step 3: 모델 컴파일

캘리브레이션 데이터셋과 모델이 준비되면 모델을 컴파일할 수 있습니다.

```bash
python model_compile.py --onnx_path {path_to_onnx_model} --calib_data_path {path_to_calibration_dataset} --save_path {path_to_save_model} --quant_percentile {quantization_percentile} --topk_ratio {topk_ratio} --inference_scheme {inference_scheme}
```

**작업 내용:**

- ONNX 모델 로드
- 캘리브레이션 데이터 로드
- 모델을 `.mxq` 형식으로 컴파일

**매개변수:**

- `--onnx_path`: ONNX 모델 경로
- `--calib_data_path`: 캘리브레이션 데이터 경로
- `--save_path`: MXQ 모델을 저장할 경로
- `--quant_percentile`: 양자화 백분위수
- `--topk_ratio`: Top-k 비율
- `--inference_scheme`: 추론 스키마(single, multi, global, global4, global8)

**출력:**

- 컴파일된 모델을 포함하는 `{path_to_save_model}` 파일 경로

예를 들어, 명령은 다음과 같습니다:

```bash
python model_compile.py --onnx_path ./resnet50.onnx --calib_data_path ./resnet50_cali --save_path ./resnet50.mxq --quant_percentile 0.9999 --topk_ratio 0.01 --inference_scheme single
```

위 명령을 실행하면 컴파일된 모델이 현재 디렉토리에 `resnet50.mxq`로 저장됩니다.
