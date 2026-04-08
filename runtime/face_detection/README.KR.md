# 얼굴 탐지 모델 추론

이 튜토리얼은 Mobilint `qbruntime`를 사용해 컴파일된 얼굴 탐지 모델로 추론을 실행하는 방법을 설명합니다.

문서 구성은 [../object_detection/README.KR.md](../object_detection/README.KR.md)를 따르되, 후처리와 라벨 체계는 단일 클래스 얼굴 탐지기에 맞게 조정했습니다.

이 가이드는 [../../compilation/face_detection/README.KR.md](../../compilation/face_detection/README.KR.md)에서 이어집니다. 아래와 같은 컴파일된 모델이 이미 준비되어 있다고 가정합니다.

- `../../compilation/face_detection/yolov12m-face.mxq`

## 사전 준비

추론을 실행하기 전에 다음이 준비되어 있어야 합니다.

- `qbruntime`
- 컴파일된 `.mxq` 얼굴 탐지 모델
- Python 패키지: `opencv-python`, `numpy`, `torch`

필요한 Python 패키지는 아래 명령으로 설치할 수 있습니다.

```bash
pip install opencv-python numpy torch
```

## 개요

추론 파이프라인은 `inference_mxq.py`에 구현되어 있으며, 다음 단계로 구성됩니다.

1. **모델 로드**: `qbruntime`로 컴파일된 `.mxq` 모델을 로드합니다.
2. **전처리**: 입력 이미지를 읽고, 컴파일 단계와 동일한 `640x640` letterbox 전처리를 적용합니다.
3. **추론**: Mobilint NPU에서 모델을 실행합니다.
4. **후처리**: YOLO 출력 head를 재정렬하고, anchorless 예측을 디코딩한 뒤, NMS를 적용합니다.
5. **시각화**: 원본 이미지 위에 얼굴 박스와 confidence score를 그립니다.

컴파일된 그래프와 출력 구조를 확인하려면 `.mblt` 파일을 [Mobilint Netron](https://netron.mobilint.com/)에서 열어볼 수 있습니다.

## 추론 실행

스크립트는 먼저 accelerator와 모델 설정을 초기화합니다.

```python
acc = qbruntime.Accelerator()
mc = qbruntime.ModelConfig()
mc.set_single_core_mode(None, [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)])

model = qbruntime.Model(args.model_path, mc)
model.launch(acc)
```

그 다음 입력 이미지를 읽고 letterbox 전처리를 적용합니다. normalization은 컴파일 단계에서 모델 안으로 융합되므로, 런타임 입력은 `UInt8` 형식을 유지합니다.

```python
def preprocess_yolo(img_path: str, img_size: tuple[int, int] = (640, 640)) -> np.ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ...
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(img)
```

이후 `YoloPostProcessAnchorless`가 다음 작업을 수행합니다.

- NPU 출력을 box head와 classification head로 분리
- distribution focal loss (DFL) 기반 박스 예측 디코딩
- confidence threshold 기반 필터링
- Non-Maximum Suppression (NMS) 적용

마지막으로 `YoloVisualizer`가 박스를 원본 이미지 크기에 맞게 다시 스케일링하고 결과 이미지를 저장합니다.

예제 실행 명령은 다음과 같습니다.

```bash
python inference_mxq.py --model-path ../../compilation/face_detection/yolov12m-face.mxq --image-path ../rc/cr7.jpg --output-path ./tmp/cr_demo.jpg --conf-thres 0.25 --iou-thres 0.45
```

이 예제는 컴파일 튜토리얼에서 만든 산출물과 맞추기 위해 모델 경로를 명시적으로 전달합니다.

## 파라미터

- `--model-path`: 컴파일된 `.mxq` 모델 파일 경로
- `--image-path`: 입력 이미지 경로
- `--output-path`: 시각화된 결과 이미지를 저장할 경로
- `--conf-thres`: 검출 결과 필터링에 사용할 confidence threshold
- `--iou-thres`: NMS에 사용할 IoU threshold

## 예상 출력

스크립트는 `tmp/cr_demo.jpg`와 같은 결과 이미지를 저장하며, 원본 이미지 위에 얼굴 bounding box를 그립니다.

이 예제는 단일 클래스 탐지 모델이므로, 유지된 모든 detection은 `face` 라벨로 표시됩니다.
