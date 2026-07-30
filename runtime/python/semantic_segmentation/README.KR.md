# 시맨틱 세그멘테이션 런타임

이 튜토리얼은 Mobilint `qbruntime`로 컴파일된 `YOLO26m-sem` MXQ 모델을 실행하고 Cityscapes 시맨틱 세그멘테이션 overlay 이미지를 저장하는 방법을 설명합니다.

시작하기 전에 [시맨틱 세그멘테이션 컴파일 튜토리얼](../../../compilation/semantic_segmentation/README.KR.md)을 완료하세요. 기본 명령은 컴파일된 모델이 `../../../compilation/semantic_segmentation/yolo26m-sem.mxq`에 있다고 가정합니다.

## 사전 준비

다음 구성 요소가 준비되어 있어야 합니다.

- Mobilint NPU 드라이버 및 `qbruntime`
- 컴파일된 `yolo26m-sem.mxq` 모델
- Python 패키지: `opencv-python`, `numpy`

추가 Python 패키지가 없다면 다음 명령으로 설치하세요.

```bash
pip install opencv-python numpy
```

Mobilint 환경에 맞는 Python 패키지 또는 시스템 패키지로 `qbruntime`을 설치하세요. 드라이버와 런타임 설정은 [Python 런타임 가이드](../README.KR.md)를 참고하세요.

## 개요

런타임 코드는 다음 단계를 수행합니다.

1. `qbruntime`로 MXQ 모델을 불러오고 실행합니다.
2. `munster.png`를 읽고 `1024x2048` YOLO 방식 letterbox 전처리를 적용합니다.
3. Mobilint NPU에서 추론을 실행합니다.
4. 19개 Cityscapes logit에 `argmax`를 적용합니다.
5. Letterbox 패딩을 제거하고 class map을 원본 이미지 크기로 복원합니다.
6. Cityscapes palette를 적용하고 원본 이미지 위에 결과를 합성합니다.

컴파일된 모델에는 `/255` 정규화가 포함되어 있으므로 런타임 입력은 `uint8` 형식을 유지합니다.

## 이 튜토리얼의 파일

- `inference_mxq.py`: 전처리, NPU 추론, 후처리, 시각화를 수행합니다.
- `postprocess.py`: Semantic logit에 `argmax`를 적용하고 원본 이미지 크기를 복원합니다.
- `visualize.py`: Cityscapes palette를 적용하고 overlay 이미지를 저장합니다.

## 입출력 Shape

컴파일된 모델이 알려주는 shape은 다음과 같습니다.

- 입력: `(1024, 2048, 3)` HWC `uint8`
- 출력: `(1024, 2048, 19)` HWC `float32` logit

ONNX 모델은 graph 안에 `argmax`가 포함되어 `(1, 1024, 2048)` class map을 반환하지만, MXQ 모델은 19개 logit을 모두 출력합니다. 따라서 런타임에서 다음 연산을 적용해야 합니다.

```python
class_map = np.argmax(logits, axis=-1)
```

MXQ 출력의 공간 크기는 ONNX class map의 공간 크기와 같으므로 별도의 output upsampling은 필요하지 않습니다.

## 전처리

`inference_mxq.py`는 모델 입력 shape을 읽고 컴파일 과정과 동일한 중앙 정렬 letterbox 변환을 적용합니다.

- 원본 이미지의 종횡비 유지
- Bilinear interpolation으로 크기 조정
- RGB 값 `(114, 114, 114)`로 `1024x2048`까지 패딩
- `uint8` 입력 유지

제공된 `munster.png`는 이미 `2048x1024` 크기이므로 기본 모델에서는 크기 조정이나 패딩이 필요하지 않습니다.

## 예제 실행

이 디렉토리에서 다음 명령을 실행하세요.

```bash
python inference_mxq.py
```

기본값은 다음과 같습니다.

- 모델: `../../../compilation/semantic_segmentation/yolo26m-sem.mxq`
- 입력 이미지: `../rc/munster.png`
- 출력 이미지: `./tmp/munster_semantic_demo.png`
- Overlay 불투명도: `0.7`

모든 경로를 직접 지정하려면 다음과 같이 실행하세요.

```bash
python inference_mxq.py \
  --model-path ../../../compilation/semantic_segmentation/yolo26m-sem.mxq \
  --image-path ../rc/munster.png \
  --output-path ./tmp/munster_semantic_demo.png \
  --overlay-alpha 0.7
```

## 파라미터

- `--model-path`: 컴파일된 `.mxq` 모델 경로
- `--image-path`: 입력 이미지 경로
- `--output-path`: 시각화된 segmentation 이미지 저장 경로
- `--overlay-alpha`: `0`에서 `1` 사이의 segmentation overlay 불투명도. 기본값: `0.7`

## 예상 출력

스크립트는 `2048x1024` 크기의 `tmp/munster_semantic_demo.png`를 저장합니다. 각 예측 class는 공식 Cityscapes 색상으로 표시됩니다.
