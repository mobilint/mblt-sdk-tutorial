# 객체 탐지 모델 컴파일

본 튜토리얼은 Mobilint `qbcompiler`를 사용하여 객체 탐지(Object Detection) 모델을 컴파일하는 상세 가이드를 제공합니다.

여기에서는 Ultralytics에서 제공하는 COCO 데이터셋 사전 학습 모델인 [YOLO11m](https://docs.ultralytics.com/models/yolo11/)을 사용합니다. 이 모델은 이미지 내의 다양한 객체를 탐지하고 위치를 식별하는 객체 탐지 모델입니다.

## 사전 요구사항

시작하기 전에 다음이 설치되어 있는지 확인하세요:

- qbcompiler SDK 컴파일러 설치 (버전 >= 0.11 필요)

또한 다음 패키지를 설치해야 합니다:

```bash
pip install ultralytics
```

## 개요

컴파일 과정은 크게 세 단계로 진행됩니다:

1. **모델 준비**: 모델을 다운로드하고 ONNX 형식으로 내보냅니다.
2. **캘리브레이션 데이터셋 준비**: COCO 데이터셋에서 샘플을 추출하여 캘리브레이션 데이터를 생성합니다.
3. **모델 컴파일**: 캘리브레이션 데이터를 활용해 모델을 `.mxq` 형식으로 변환합니다.

## Step 1: 모델 준비

먼저 모델을 준비해야 합니다. `ultralytics` 라이브러리를 사용하여 사전 훈련된 모델을 다운로드하고 ONNX 형식으로 내보냅니다.

```bash
yolo export model=yolo11m.pt format=onnx # 모델을 ONNX 형식으로 내보내기
```

실행 후, 내보낸 ONNX 모델은 현재 디렉토리에 `yolo11m.onnx`로 저장됩니다.

## Step 2: 캘리브레이션 데이터셋 준비

캘리브레이션 데이터셋은 모델의 실제 입력 분포를 대표하는 이미지들로 구성됩니다. YOLO11m 모델은 [COCO 데이터셋](https://cocodataset.org/#download)으로 학습되었으므로, COCO 샘플을 사용하여 캘리브레이션을 진행합니다.

HuggingFace 계정이 필요하며, 로그인이 되어 있지 않다면 다음 명령어로 로그인하세요:

```bash
hf auth login --token <your_huggingface_token>
```

제공된 `prepare_coco.py` 스크립트를 사용하면 COCO 데이터셋의 이미지 URL을 읽어 랜덤하게 샘플을 선택하고 `coco-selected` 디렉토리에 자동으로 다운로드합니다.

```bash
python prepare_coco.py
```

**주요 작업:**
- HuggingFace에서 COCO 이미지 URL 데이터셋 로드
- 이미지를 랜덤하게 선택하여 캘리브레이션 데이터셋 구성
- 선택된 이미지를 `coco-selected` 디렉토리에 저장

**결과물:**
- `coco-selected`: 캘리브레이션용 이미지 데이터셋

## Step 3: 모델 컴파일

컴파일을 진행하기 전, 모델에 필요한 전처리 단계를 확인해야 합니다. YOLO 모델은 주로 `LetterBox` 전처리를 사용하며, 이에 대한 상세 내용은 [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)에서 확인할 수 있습니다.

Mobilint 컴파일 API는 이러한 전처리 과정을 내부적으로 수행하며, 전처리 연산을 MXQ 모델에 통합(fuse)하여 NPU 연산 효율을 극대화합니다.

`model_compile.py`에서는 다음과 같이 전처리 파이프라인을 정의합니다. 이 설정은 캘리브레이션 과정에서 사용되며, 전처리 모듈이 모델에 통합됩니다.

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

또한, 다음과 같이 입력 프로세스 및 양자화 설정을 정의합니다.

```python
input_process_config = InputProcessConfig(
    uint8_input=Uint8InputConfig(apply=True, inputs=[]), # uint8 입력 사용
    image_channels=3,
    preprocessing=preprocessing_config,
)

quantization_config = QuantizationConfig.from_kwargs(
    quantization_method=1,  # 0: per tensor, 1: per channel
    quantization_output=1,  # 0: layer, 1: channel
    quantization_mode=2,    # maxpercentile
    percentile=0.999,
    topk_ratio=0.01,
)
```

설정이 완료되면 다음과 같은 명령어로 컴파일을 실행할 수 있습니다.

```bash
python model_compile.py --onnx-path {path_to_onnx_model} --calib-data-path {path_to_calibration_dataset} --save-path {path_to_save_model} --quant-percentile {quantization_percentile} --topk-ratio {topk_ratio} --inference-scheme {inference_scheme}
```

**주요 작업:**
- ONNX 모델 로드
- 캘리브레이션 데이터 로드
- 모델을 `.mxq` 형식으로 컴파일

**파라미터:**
- `--onnx-path`: ONNX 모델 경로
- `--calib-data-path`: 캘리브레이션 데이터 경로
- `--save-path`: MXQ 모델 저장 경로
- `--quant-percentile`: 양자화 백분위수
- `--topk-ratio`: Top-K 비율
- `--inference-scheme`: 추론 방식 (Core 할당 전략)

**추론 방식(Inference Scheme):**
모델의 코어 활용 전략을 지정하는 옵션입니다:
- `single`: 단일 코어 사용
- `multi`: 다중 코어 분산 처리
- `global4` / `global8`: 4개 또는 8개 코어를 연합하여 처리

자세한 내용은 [다중 코어 모드](https://docs.mobilint.com/v0.29/en/multicore.html) 문서를 참고하세요.

**실행 예시:**
```bash
python model_compile.py --onnx-path ./yolo11m.onnx --calib-data-path ./coco-selected --save-path ./yolo11m.mxq --quant-percentile 0.999 --topk-ratio 0.001 --inference-scheme single
```

위 명령어를 실행하면 현재 디렉토리에 `yolo11m.mxq` 파일이 생성됩니다.
