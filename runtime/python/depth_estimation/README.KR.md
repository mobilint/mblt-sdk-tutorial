# 깊이 추정 런타임

이 튜토리얼은 Mobilint `qbruntime`로 컴파일된 `YOLO26m-depth` MXQ 모델을 실행하고 컬러 depth overlay 이미지를 저장하는 방법을 설명합니다.

시작하기 전에 [../../../compilation/depth_estimation/README.KR.md](../../../compilation/depth_estimation/README.KR.md)의 컴파일 과정을 완료하세요. 기본 런타임 명령은 컴파일된 모델이 `../../../compilation/depth_estimation/yolo26m-depth.mxq`에 있다고 가정합니다.

## 사전 준비

다음 구성 요소가 준비되어 있어야 합니다.

- Mobilint NPU 드라이버 및 `qbruntime`
- 컴파일된 `yolo26m-depth.mxq` 모델
- Python 패키지: `opencv-python`, `numpy`, `torch`

추가 Python 패키지가 없다면 다음 명령으로 설치하세요.

```bash
pip install opencv-python numpy torch
```

NPU 드라이버와 `qbruntime` 설정은 [Python 런타임 가이드](../README.KR.md)를 참고하세요.

## 개요

`inference_mxq.py`의 런타임 흐름은 다음 단계를 수행합니다.

1. `qbruntime`로 MXQ 모델을 로드하고 실행합니다.
2. RGB 이미지를 읽고 `768x768` YOLO 방식 letterbox 전처리를 적용합니다.
3. 모델이 알려주는 HWC 또는 CHW 입력 레이아웃에 맞춥니다.
4. Mobilint NPU에서 추론을 실행합니다.
5. ONNX 출력 shape과 맞도록 1/4 크기의 MXQ 출력을 4배 upsampling합니다.
6. Letterbox 패딩을 제거하고 depth map을 원본 이미지 크기로 조정합니다.
7. Inverse depth를 컬러로 변환해 원본 이미지 위에 합성합니다.

컴파일된 MXQ 모델에는 `/255` 정규화가 포함되어 있으므로 런타임 입력은 `uint8` 형식을 유지합니다.

## 이 튜토리얼의 파일

- `inference_mxq.py`: 전처리, NPU 추론, 후처리, 시각화를 실행합니다.
- `postprocess.py`: MXQ 출력 레이아웃을 정리하고 upsampling한 뒤 원본 이미지 크기로 복원합니다.
- `visualize.py`: Inverse depth를 JET 컬러맵으로 변환하고 overlay 이미지를 저장합니다.

## 전처리

스크립트는 모델 입력 shape을 읽고 컴파일 과정과 동일한 letterbox 변환을 적용합니다. 기본 모델은 `(768, 768, 3)` HWC 입력을 사용합니다.

```python
model_input, borders = preprocess_yolo(image_rgb, input_shape)
outputs = model.infer([model_input])
```

반환된 패딩 크기는 후처리 단계로 전달되어 패딩 영역을 정확하게 제거하는 데 사용됩니다.

## 필수 MXQ 출력 Upsampling

ONNX 모델은 `(1, 1, 768, 768)` shape의 depth tensor를 출력하지만, 컴파일된 MXQ 모델의 출력 shape은 `(1, 1, 192, 192)`입니다. 따라서 letterbox를 복원하기 전에 MXQ 출력을 bilinear 방식으로 4배 upsampling해야 합니다.

```python
depth = F.interpolate(
    depth,
    scale_factor=4.0,
    mode="bilinear",
    align_corners=False,
)
```

이 연산이 끝나면 `postprocess.py`가 출력 shape이 `768x768`인지 확인하고 letterbox 패딩을 제거한 뒤, depth map을 원본 이미지 크기로 조정합니다.

## 예제 실행

이 디렉토리에서 다음 명령을 실행하세요.

```bash
python inference_mxq.py
```

기본값은 다음과 같습니다.

- 모델: `../../../compilation/depth_estimation/yolo26m-depth.mxq`
- 입력 이미지: `../rc/bus.jpg`
- 출력 이미지: `./tmp/bus_depth_demo.jpg`
- Depth overlay 불투명도: `0.7`

경로와 불투명도를 직접 지정하려면 다음과 같이 실행하세요.

```bash
python inference_mxq.py \
  --model-path ../../../compilation/depth_estimation/yolo26m-depth.mxq \
  --image-path ../rc/bus.jpg \
  --output-path ./tmp/bus_depth_demo.jpg \
  --overlay-alpha 0.7
```

## 파라미터

- `--model-path`: 컴파일된 `.mxq` 모델 경로
- `--image-path`: 입력 이미지 경로
- `--output-path`: 시각화된 depth 이미지 저장 경로
- `--overlay-alpha`: `0`에서 `1` 사이의 depth map 불투명도. 기본값: `0.7`

## 예상 출력

스크립트는 `tmp/bus_depth_demo.jpg`를 저장합니다. 가까운 영역은 따뜻한 색으로, 먼 영역은 차가운 색으로 표시됩니다.
