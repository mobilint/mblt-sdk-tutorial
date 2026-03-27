# 인스턴스 분할 모델 컴파일

본 튜토리얼은 모빌린트 `qbcompiler`를 사용하여 인스턴스 분할 모델을 컴파일하는 방법에 대한 포괄적인 가이드를 제공합니다.

Ultralytics에서 만든 COCO 데이터셋으로 사전 학습된 [YOLO11m-seg](https://docs.ultralytics.com/models/yolo11/) 모델을 사용할 것입니다. 이 모델은 인스턴스 분할을 수행하여 이미지 내의 개별 객체를 식별하고 마스크합니다.

## 사전 준비

시작하기 전에 다음이 설치되어 있는지 확인하세요:

- qbcompiler v1.0.0
- COCO 데이터셋에 접근 가능한 HuggingFace 계정 (gated 데이터셋 사용을 위해)

또한, 다음 패키지들을 설치해야 합니다:

```bash
pip install ultralytics aiohttp aiofiles
```

## 개요

컴파일 워크플로우는 세 가지 주요 단계로 구성됩니다:

1. **모델 준비**: 모델을 다운로드하고 ONNX 형식으로 내보냅니다.
2. **캘리브레이션 데이터셋 준비**: COCO에서 대표적인 캘리브레이션 데이터셋을 생성합니다.
3. **모델 컴파일**: 캘리브레이션 데이터를 사용하여 모델을 `.mxq` 형식으로 변환합니다.

## 단계 1: 모델 준비

먼저 모델을 준비해야 합니다. `ultralytics` 라이브러리를 사용하여 사전 학습된 모델을 다운로드하고 ONNX 형식으로 내보냅니다.

```bash
yolo export model=yolo11m-seg.pt format=onnx # 모델을 ONNX 형식으로 내보내기
```

실행 후, 내보낸 ONNX 모델은 현재 디렉토리에 `yolo11m-seg.onnx`로 저장됩니다.

캘리브레이션 데이터셋은 모델의 전형적인 입력 분포를 나타내는 이미지들의 집합으로 구성됩니다. YOLO11m은 [COCO 데이터셋](https://cocodataset.org/#download)을 기반으로 학습되었으므로 캘리브레이션에 COCO 샘플을 사용할 것입니다.

데이터셋을 사용하기 전에 [HuggingFace](https://huggingface.co/)에 가입하세요. 그 다음, 아래 명령어를 사용하여 HuggingFace에 로그인하고 `<your_huggingface_token>`을 실제 HuggingFace 토큰으로 변경하세요:

```bash
hf auth login --token <your_huggingface_token>
```

HuggingFace 토큰을 모르는 경우, [HuggingFace 계정 설정](https://huggingface.co/settings/tokens)에서 확인할 수 있습니다.

과정을 자동화하기 위해 `prepare_coco.py` 스크립트를 사용합니다. 이 스크립트는 COCO 데이터셋에서 URL을 읽어 무작위로 선택하고, `coco-selected` 디렉토리로 이미지를 다운로드합니다.

```bash
python prepare_coco.py
```

**수행 작업:**

- HuggingFace에서 COCO 이미지 URL 다운로드
- 캘리브레이션 데이터셋을 구성하기 위해 이미지를 무작위로 선택
- `coco-selected` 디렉토리에 이미지 저장

**출력:**

- `coco-selected`: 캘리브레이션 데이터셋

선택한 이미지 데이터셋이 우리가 사용할 캘리브레이션 데이터셋입니다.

컴파일을 실행하기 전에 필요한 전처리 단계를 확인하세요.

[Ultralytics GitHub](https://github.com/ultralytics/ultralytics)에 자세히 설명되어 있듯이 YOLO 모델은 일반적으로 `LetterBox` 연산을 사용합니다.

모빌린트 컴파일 API는 이러한 전처리 단계를 내부적으로 수행하며, NPU 효율성을 극대화하기 위해 작동을 MXQ 모델에 직접 융합합니다.

`model_compile.py`에서 아래와 같이 전처리 파이프라인을 정의합니다. 이 파이프라인은 캘리브레이션에 사용되며, 정규화 모듈을 딥러닝 모델에 융합할 것입니다.

```python
preprocess_pipeline = [
    {
    "op": "letterbox",
    "height": 640,
    "width": 640,
    "padValue": 114
    }
]
preprocessing_config = PreprocessingConfig(
    apply=True,
    auto_convert_format=True,
    pipeline=preprocess_pipeline,
    input_configs={},
)
```

또한, 다음 전처리 구성 및 양자화 구성을 정의합니다.

```python
calibration_config = CalibrationConfig(
        method=1,  # 0: 텐서, 1: 채널
        output=1,  # 0: 레이어, 1: 채널
        mode=1,  # maxpercentile
        max_percentile={
            "percentile": 0.9999,  # quantization percentile
            "topk_ratio": 0.01,  # quantization topk
        },
    )
```

설정을 구성한 후, 코드를 다음과 같이 실행할 수 있습니다.

```bash
python model_compile.py --onnx-path {path_to_onnx_model} --calib-data-path {path_to_calibration_dataset} --save-path {path_to_save_model}
```

**수행 작업:**

- ONNX 모델 로드
- 캘리브레이션 데이터 로드
- `.mxq` 형식으로 모델 컴파일

**파라미터:**

- `--onnx-path`: ONNX 모델의 경로
- `--calib-data-path`: 캘리브레이션 데이터의 경로
- `--save-path`: MXQ 모델을 저장할 경로

**출력:**

- 컴파일된 모델을 포함하는 `{path_to_save_model}` 파일 경로

예시 명령어는 다음과 같습니다:

```bash
python model_compile.py --onnx-path ./yolo11m-seg.onnx --calib-data-path ./coco-selected --save-path ./yolo11m-seg.mxq 
```

위의 명령어를 실행한 후, 컴파일된 모델은 현재 디렉토리에 `yolo11m-seg.mxq`로 저장됩니다.
