# 회전 바운딩 박스 모델 컴파일

이 튜토리얼은 Mobilint `qbcompiler`를 사용하여 회전 바운딩 박스(OBB) 탐지 모델을 컴파일하는 방법을 설명합니다.

이 예제에서는 Ultralytics의 [YOLO11m-obb](https://docs.ultralytics.com/tasks/obb/) 모델을 사용합니다. 이 모델은 회전 객체 탐지를 위해 학습되었으며, DOTA와 같은 항공 영상 데이터셋에서 자주 사용됩니다.

## 사전 준비

시작하기 전에 다음 항목이 준비되어 있어야 합니다:

- `qbcompiler`
- Python 3.10 이상

튜토리얼 의존성은 다음과 같이 설치합니다:

```bash
pip install ultralytics
```

데이터셋 준비 스크립트는 Python 표준 라이브러리만 사용합니다.

## 개요

컴파일 워크플로우는 다음 세 단계로 구성됩니다:

1. **모델 준비**: 사전 학습된 OBB 모델을 ONNX로 내보냅니다.
2. **캘리브레이션 데이터셋 준비**: DOTAv1을 다운로드하고 캘리브레이션 이미지를 선택합니다.
3. **모델 컴파일**: ONNX 모델을 `.mxq` 형식으로 변환합니다.

## 단계 1: 모델 준비

사전 학습된 YOLO11 OBB 모델을 ONNX로 내보냅니다:

```bash
yolo export model=yolo11m-obb.pt format=onnx
```

내보내기가 완료되면 ONNX 모델은 현재 디렉토리에 `yolo11m-obb.onnx`로 저장됩니다.

## 단계 2: 캘리브레이션 데이터셋 준비

캘리브레이션 데이터셋은 모델이 실제로 받게 될 입력 분포를 반영해야 합니다. `yolo11m-obb.pt`는 DOTA 스타일의 항공 영상에 맞춰 학습되었으므로, 이 튜토리얼에서는 DOTAv1을 캘리브레이션 데이터로 사용합니다.

Ultralytics에서 제공하는 아카이브를 바로 사용하여 데이터셋을 준비할 수 있습니다:

```bash
python prepare_dota.py
```

이 스크립트는 다음 작업을 수행합니다:

- `DOTAv1.zip`이 없으면 다운로드합니다.
- 아카이브를 `./DOTAv1`에 압축 해제합니다.
- 고정된 시드로 100장의 이미지를 무작위 선택합니다.
- 선택한 이미지를 `./dota-selected`로 복사합니다.

생성된 `dota-selected` 디렉토리가 `model_compile.py`에서 사용하는 캘리브레이션 데이터셋입니다.

### 선택 가능한 데이터셋 인자

이미 데이터셋을 수동으로 다운로드한 경우에는 기존 파일을 재사용할 수 있습니다:

```bash
python prepare_dota.py --skip-download --zip-path ./DOTAv1.zip
```

이미 압축까지 풀어 두었고 캘리브레이션 서브셋만 만들고 싶다면 다음과 같이 실행합니다:

```bash
python prepare_dota.py --skip-download --extract-dir ./DOTAv1 --output-dir ./dota-selected --num-images 100
```

## 단계 3: 모델 컴파일

컴파일 전에 내보낸 모델에 필요한 전처리를 확인해야 합니다. Ultralytics OBB 모델은 letterbox 리사이즈를 사용하며, 이 튜토리얼도 캘리브레이션 시 동일한 동작을 맞춰 사용합니다.

`model_compile.py`에서는 전처리 파이프라인을 다음과 같이 설정합니다:

```python
preprocess_pipeline = [{"op": "letterbox", "height": 1024, "width": 1024, "padValue": 114}]
preprocessing_config = PreprocessingConfig(
    apply=True,
    auto_convert_format=True,
    pipeline=preprocess_pipeline,
    input_configs={},
)
```

Mobilint 컴파일 API는 캘리브레이션 과정에서 이 파이프라인을 적용합니다. `/255` 정규화는 `Uint8InputConfig`를 통해 MXQ 모델에 융합되므로 런타임에서는 `uint8` 입력을 직접 전달할 수 있습니다. 반면 letterbox 같은 공간 변환은 융합되지 않으므로 런타임에서 계속 적용해야 합니다.

양자화 설정은 다음과 같습니다:

```python
calibration_config = CalibrationConfig(
    method=1,
    output=1,
    mode=1,
    max_percentile={
        "percentile": 0.9999,
        "topk_ratio": 0.01,
    },
)
```

### ARIES

이 튜토리얼 스크립트는 `aries2`를 대상으로 컴파일하며, 하나의 MXQ 파일에 여러 추론 스킴을 담기 위해 `inference_scheme="all"`을 사용합니다.

다음 명령으로 컴파일을 실행합니다:

```bash
python model_compile.py --onnx-path ./yolo11m-obb.onnx --calib-data-path ./dota-selected --save-path ./yolo11m-obb.mxq
```

명령이 완료되면:

- `yolo11m-obb.mxq`가 현재 디렉토리에 저장됩니다.
- 중간 그래프 파일인 `yolo11m-obb.mblt`가 ONNX 파일 옆에 생성됩니다.

## 이 튜토리얼의 파일

- `model_compile.py`: ONNX 모델을 ARIES2용 MXQ로 컴파일합니다.
- `prepare_dota.py`: DOTAv1을 다운로드하거나 재사용하고 캘리브레이션 이미지를 준비합니다.
- `README.md`: 영어 튜토리얼 문서입니다.
- `README.KR.md`: 한국어 튜토리얼 문서입니다.
