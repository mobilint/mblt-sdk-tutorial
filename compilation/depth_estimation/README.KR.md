# 깊이 추정 모델 컴파일

이 튜토리얼은 Mobilint `qbcompiler`를 사용해 Ultralytics `YOLO26m-depth` 모델을 `.mxq` 및 `.mblt` 파일로 컴파일하는 방법을 설명합니다.

이 모델은 RGB 입력을 사용하며 `768x768` 크기의 YOLO 방식 letterbox 전처리를 적용합니다. 캘리브레이션 이미지는 [Ultralytics 데이터셋 압축 파일](https://github.com/ultralytics/assets/releases/download/v0.0.0/nyu-depth.zip)에 포함된 NYU Depth V2 검증 데이터에서 선택합니다.

## 사전 준비

시작하기 전에 `qbcompiler`와 이 튜토리얼에서 사용하는 Python 패키지를 설치하세요.

```bash
pip install ultralytics opencv-python
```

압축된 데이터셋 파일의 크기는 약 502 MB이며, 압축을 풀면 약 1.5 GB의 공간을 사용합니다.

## 개요

워크플로우는 세 단계로 구성됩니다.

1. `YOLO26m-depth` 모델을 ONNX로 export합니다.
2. NYU Depth V2를 다운로드하고 캘리브레이션용 검증 이미지를 선택합니다.
3. `768x768` letterbox 전처리를 적용해 ONNX 모델을 컴파일합니다.

## 1단계: 모델 준비

사전 학습된 모델을 ONNX로 export합니다.

```bash
yolo export model=yolo26m-depth.pt format=onnx
```

명령이 끝나면 현재 디렉토리에 `yolo26m-depth.onnx`가 생성됩니다. 제공된 `yolo26m-depth.onnx`가 이미 이 디렉토리에 있다면 그대로 사용할 수 있습니다.

## 2단계: 캘리브레이션 데이터셋 준비

`prepare_nyu_depth_v2.py`는 [Ultralytics NYU Depth V2 압축 파일](https://github.com/ultralytics/assets/releases/download/v0.0.0/nyu-depth.zip)을 다운로드하고 압축을 풉니다. 그다음 `nyu-depth/images/val`에서 RGB 이미지를 재현 가능한 방식으로 선택해 캘리브레이션 데이터셋을 만듭니다. 컴파일러는 RGB 모델 입력만 사용하므로 depth map은 복사하지 않습니다.

기본적으로 스크립트는 임시 디렉토리에 압축 파일을 다운로드하고 압축을 풉니다. 이후 seed `42`를 사용해 검증 이미지 100장을 선택하고 `./nyu-depth-selected`에 복사합니다. 임시 압축 파일과 전체 데이터셋은 작업이 끝나면 자동으로 삭제됩니다.

```bash
python prepare_nyu_depth_v2.py
```

선택할 이미지 수나 출력 디렉토리, seed를 변경할 수 있습니다.

```bash
python prepare_nyu_depth_v2.py --num-images 200 --output-dir ./nyu-depth-calibration --seed 7
```

출력 디렉토리가 이미 존재한다면 `--overwrite`를 추가해 기존 디렉토리를 교체하세요.

## 2-1단계 (선택): 이미지를 전처리된 Tensor로 변환

일반적으로 `qbcompiler`는 raw 이미지 디렉토리를 입력으로 받아 캘리브레이션 과정에서 전처리 설정을 적용할 수 있습니다. 명시적으로 전처리된 재사용 가능한 캘리브레이션 텐서가 필요할 때만 이 선택 단계를 사용하세요.

`convert_img_to_tensor.py`는 컴파일러 경로와 동일하게 다음 전처리를 적용합니다.

- BGR에서 RGB로 변환
- 종횡비를 유지하는 `768x768` letterbox 리사이즈
- 값 `114`를 사용하는 상수 패딩
- `/255` 연산을 적용한 `float32` HWC 텐서 변환

```bash
python convert_img_to_tensor.py
```

스크립트는 텐서를 `calib_data_tensor`에 저장합니다. 이 텐서를 `mxq_compile`에서 사용하려면 raw 이미지 캘리브레이션 경로를 텐서 디렉토리로 바꾸고 내장 전처리 설정을 제외해야 합니다. 제공된 `model_compile.py`는 raw 이미지 워크플로우를 사용합니다.

## 3단계: 모델 컴파일

`model_compile.py`는 raw RGB 캘리브레이션 이미지에 다음 전처리 설정을 적용합니다.

```python
preprocessing_config = PreprocessingConfig(
    apply=True,
    auto_convert_format=True,
    pipeline=[{"op": "letterbox", "height": 768, "width": 768, "padValue": 114}],
    input_configs={},
)
```

`/255` 정규화는 `Uint8InputConfig`를 통해 MXQ 모델에 융합되므로, 컴파일된 모델은 `uint8` 입력을 받습니다. Letterbox는 공간 변환이므로 추론 전에 계속 적용해야 합니다.

대상 NPU에 맞춰 모델을 컴파일하세요.

```bash
# ARIES
python model_compile.py --target-device aries-rb

# REGULUS
python model_compile.py --target-device regulus-rb
```

명령을 실행하면 다음 파일이 생성됩니다.

- `yolo26m-depth.mxq`: NPU에서 실행할 양자화 모델
- `yolo26m-depth.mblt`: 검사에 사용할 중간 그래프

모든 경로는 필요에 따라 변경할 수 있습니다.

```bash
python model_compile.py \
  --onnx-path ./yolo26m-depth.onnx \
  --calib-data-path ./nyu-depth-selected \
  --save-path ./yolo26m-depth.mxq \
  --mblt-path ./yolo26m-depth.mblt \
  --target-device aries-rb
```
