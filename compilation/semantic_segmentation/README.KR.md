# 시맨틱 세그멘테이션 모델 컴파일

이 튜토리얼은 Mobilint `qbcompiler`를 사용해 Ultralytics `YOLO26m-sem` Cityscapes 시맨틱 세그멘테이션 모델을 `.mxq` 및 `.mblt` 파일로 컴파일하는 방법을 설명합니다.

이 모델은 RGB 입력을 사용하며 `1024x2048` 크기의 YOLO 방식 letterbox 전처리를 적용합니다. 캘리브레이션 이미지는 [`Chris1/cityscapes_segmentation`](https://huggingface.co/datasets/Chris1/cityscapes_segmentation)의 `validation` split에 있는 `image` column에서 선택합니다.

## 사전 준비

시작하기 전에 `qbcompiler`와 이 튜토리얼에서 사용하는 Python 패키지를 설치하세요.

```bash
pip install ultralytics datasets opencv-python
```

이 공개 데이터셋은 인증이 필요하지 않습니다. Validation split에는 `1024x2048` 크기의 이미지와 semantic mask 쌍 500개가 있으며, Parquet 형식으로 약 1.2 GB를 사용합니다. Hugging Face `datasets`는 다운로드한 파일을 캐시에 저장하므로 충분한 디스크 공간을 준비하세요.

## 개요

1. `YOLO26m-sem` 모델을 ONNX로 export합니다.
2. Cityscapes validation split에서 캘리브레이션 이미지를 선택합니다.
3. `1024x2048` letterbox 전처리를 적용해 ONNX 모델을 컴파일합니다.

## 1단계: 모델 준비

사전 학습된 모델을 ONNX로 export합니다.

```bash
yolo export model=yolo26m-sem.pt format=onnx imgsz=1024,2048
```

명령이 끝나면 현재 디렉토리에 `yolo26m-sem.onnx`가 생성됩니다. 제공된 `yolo26m-sem.onnx`가 이미 이 디렉토리에 있다면 그대로 사용할 수 있습니다.

제공된 ONNX 모델의 입출력 shape은 다음과 같습니다.

- 입력: `[1, 3, 1024, 2048]`
- 출력: `[1, 1024, 2048]`

## 2단계: 캘리브레이션 데이터셋 준비

`prepare_cityscapes.py`는 Hugging Face Dataset Viewer에서 데이터셋의 Parquet 파일 목록을 조회하고 `validation` shard URL만 `load_dataset`에 전달합니다. `train` 및 `test` Parquet 파일은 다운로드하지 않습니다. 또한 loader는 `image` column만 읽으며, 컴파일 과정에서는 RGB 모델 입력만 캘리브레이션하므로 semantic mask는 저장하지 않습니다.

Validation split은 총 약 1.2 GB인 Parquet 파일 두 개로 구성됩니다. 기본적으로 스크립트는 seed `42`를 사용해 validation 이미지 500장 중 100장을 선택하고 `./cityscapes-selected`에 저장합니다.

```bash
python prepare_cityscapes.py
```

이미지 수, 출력 디렉토리, seed를 변경할 수 있습니다.

```bash
python prepare_cityscapes.py \
  --num-images 200 \
  --output-dir ./cityscapes-calibration \
  --seed 7
```

출력 디렉토리가 이미 존재한다면 `--overwrite`를 추가해 기존 디렉토리를 교체하세요.

## 2-1단계 (선택): 이미지를 전처리된 Tensor로 변환

일반적으로 `qbcompiler`는 raw 이미지 디렉토리를 입력으로 받아 캘리브레이션 과정에서 전처리를 적용할 수 있습니다. 명시적으로 전처리된 재사용 가능한 캘리브레이션 tensor가 필요할 때만 이 선택 단계를 사용하세요.

`convert_img_to_tensor.py`는 다음 전처리를 적용합니다.

- BGR에서 RGB로 변환
- 종횡비를 유지하는 `1024x2048` letterbox 리사이즈
- 값 `114`를 사용하는 상수 패딩
- `/255` 연산을 적용한 `float32` HWC tensor 변환

```bash
python convert_img_to_tensor.py
```

스크립트는 tensor를 `calib_data_tensor`에 저장합니다. 이 tensor로 컴파일하려면 해당 디렉토리를 `calib_data_path`로 사용하고 내장 전처리 설정을 제외해야 합니다. 제공된 `model_compile.py`는 raw 이미지 워크플로우를 사용합니다.

## 3단계: 모델 컴파일

`model_compile.py`는 raw RGB 캘리브레이션 이미지에 다음 전처리 설정을 적용합니다.

```python
preprocessing_config = PreprocessingConfig(
    apply=True,
    auto_convert_format=True,
    pipeline=[
        {
            "op": "letterbox",
            "height": 1024,
            "width": 2048,
            "padValue": 114,
        }
    ],
    input_configs={},
)
```

`/255` 정규화는 `Uint8InputConfig`를 통해 MXQ 모델에 융합되므로, 컴파일된 모델은 `uint8` 입력을 받습니다. Letterbox는 공간 변환이므로 런타임 추론 전에 계속 적용해야 합니다.

`model_compile.py`는 `torch.cuda.is_available()`이 참이면 CUDA를 사용해 MXQ를 컴파일합니다. CPU 전용 `qbcompiler` 이미지에서는 자동으로 CPU 컴파일을 선택합니다. 선택한 host device는 컴파일을 시작하기 전에 출력됩니다.

대상 NPU에 맞춰 모델을 컴파일하세요.

```bash
# ARIES
python model_compile.py --target-device aries-rb

# REGULUS
python model_compile.py --target-device regulus-rb
```

명령을 실행하면 다음 파일이 생성됩니다.

- `yolo26m-sem.mxq`: NPU에서 실행할 양자화 모델
- `yolo26m-sem.mblt`: 검사에 사용할 중간 그래프

모든 경로는 필요에 따라 변경할 수 있습니다.

```bash
python model_compile.py \
  --onnx-path ./yolo26m-sem.onnx \
  --calib-data-path ./cityscapes-selected \
  --save-path ./yolo26m-sem.mxq \
  --mblt-path ./yolo26m-sem.mblt \
  --target-device aries-rb
```
