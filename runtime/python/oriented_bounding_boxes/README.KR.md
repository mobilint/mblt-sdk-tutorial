# 회전 바운딩 박스 런타임

이 튜토리얼은 Mobilint `qbruntime`를 사용해 컴파일된 `YOLO11m-obb` MXQ 모델을 실행하는 방법을 설명합니다.

시작하기 전에 [../../../compilation/oriented_bounding_boxes/README.KR.md](../../../compilation/oriented_bounding_boxes/README.KR.md)의 컴파일 과정을 먼저 완료하세요. 이 디렉토리의 런타임 예제는 컴파일된 모델이 `../../../compilation/oriented_bounding_boxes/yolo11m-obb.mxq`에 있다고 가정합니다.

## 사전 준비

다음 구성 요소가 준비되어 있어야 합니다.

- Mobilint `qbruntime`
- 컴파일된 `.mxq` 모델 파일
- Python 패키지: `opencv-python`, `numpy`, `torch`

Python 패키지가 아직 설치되어 있지 않다면 다음 명령으로 설치할 수 있습니다.

```bash
pip install opencv-python numpy torch
```

## 개요

런타임 흐름은 `inference_mxq.py`에 구현되어 있으며 다음 단계를 따릅니다.

1. `qbruntime`로 컴파일된 MXQ 모델을 로드합니다.
2. 입력 이미지를 읽고 컴파일 단계와 동일한 `1024x1024` letterbox 전처리를 적용합니다.
3. 모델이 HWC 또는 CHW 입력 중 무엇을 기대하는지 확인해 그 형식에 맞게 자동으로 입력을 구성합니다.
4. Mobilint NPU에서 추론을 실행합니다.
5. 회전 박스를 디코드하고 rotated NMS를 적용한 뒤, DOTA 클래스 라벨과 함께 폴리곤을 렌더링합니다.

컴파일된 MXQ 모델에는 `Uint8InputConfig`를 통해 이미 `/255` 정규화가 포함되어 있으므로, 이 예제는 런타임 입력을 `uint8` 형식으로 유지합니다.

## 이 튜토리얼의 파일

- `inference_mxq.py`: 전체 추론 파이프라인을 실행하고 시각화된 결과를 저장합니다.
- `postprocess.py`: OBB 출력을 재배열하고 `cx, cy, w, h, angle` 형식으로 디코드한 뒤 rotated NMS를 적용합니다.
- `visualize.py`: 검출 결과를 원본 이미지 좌표계로 복원하고 회전 폴리곤을 그립니다.
- `dota.py`: DOTAv1 클래스 이름과 색상 정보를 제공합니다.
- `utils.py`: DFL, 회전 박스 디코드, 좌표 복원, rotated NMS 보조 함수를 제공합니다.

## 스크립트 동작 방식

스크립트는 먼저 accelerator를 초기화하고 컴파일된 모델을 실행합니다.

```python
acc = qbruntime.Accelerator()
model_config = qbruntime.ModelConfig()
model_config.set_single_core_mode(
    None,
    [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)],
)

model = qbruntime.Model(args.model_path, model_config)
model.launch(acc)
```

그 다음 이미지를 읽고 BGR을 RGB로 변환한 뒤 `1024x1024` letterbox 전처리를 적용합니다. 이 과정에서 모델 입력 shape를 확인하여 HWC 또는 CHW 형식 중 필요한 쪽으로 자동 변환합니다.

```python
def preprocess_yolo_obb(img: np.ndarray, input_shape: tuple[int, ...]) -> np.ndarray:
    if input_shape[-1] == 3:
        target_h, target_w, is_hwc = input_shape[0], input_shape[1], True
    else:
        target_h, target_w, is_hwc = input_shape[1], input_shape[2], False

    ...
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    if not is_hwc:
        img = np.transpose(img, (2, 0, 1))

    return np.expand_dims(img, 0).astype(np.uint8, copy=False)
```

추론 후에는 필요할 경우 HWC 형태의 NPU 출력을 BCHW로 다시 변환해 `postprocess.py`가 하나의 레이아웃만 처리하도록 맞춥니다. 이후 `YoloObbPostProcess`가 detection, class, angle head를 묶고 회전 박스를 디코드한 뒤 rotated NMS를 적용합니다.

최종 검출 결과는 `cx, cy, w, h, conf, cls, angle` 형식을 따르며, `visualize.py`는 이를 원본 이미지 위의 회전 폴리곤으로 변환해 표시합니다.

출력 텐서 구조나 후처리 가정을 확인하고 싶다면, 컴파일 과정에서 생성된 `.mblt` 파일을 [Mobilint Netron](https://netron.mobilint.com/)에서 열어볼 수 있습니다.

## 예제 실행

기본 샘플 경로를 그대로 사용하려면 다음 명령을 실행하세요.

```bash
python inference_mxq.py
```

이 명령은 다음 기본값을 사용합니다.

- 모델: `../../../compilation/oriented_bounding_boxes/yolo11m-obb.mxq`
- 입력 이미지: `../rc/airport.jpg`
- 출력 이미지: `./tmp/airport_demo.jpg`

경로를 명시적으로 지정하거나 임곗값을 조정하려면 다음과 같이 실행하세요.

```bash
python inference_mxq.py --model-path ../../../compilation/oriented_bounding_boxes/yolo11m-obb.mxq --image-path ../rc/airport.jpg --output-path ./tmp/airport_custom.jpg --conf-thres 0.3 --iou-thres 0.5
```

## 파라미터

- `--model-path`: 컴파일된 `.mxq` 모델 경로
- `--image-path`: 입력 이미지 경로
- `--output-path`: 시각화된 결과 이미지를 저장할 경로
- `--conf-thres`: 검출 결과를 유지할 confidence threshold. 기본값: `0.25`
- `--iou-thres`: rotated NMS에 사용할 IoU threshold. 기본값: `0.45`

## 예상 출력

스크립트는 `tmp/airport_demo.jpg`와 같은 결과 이미지를 저장하며, 원본 이미지 위에 회전 폴리곤, DOTAv1 클래스 라벨, confidence score를 그립니다.

## 참고

- 이 튜토리얼은 로컬 후처리 구현이 가정하는 `YOLO11m-obb` 출력 레이아웃을 대상으로 합니다.
- 후처리 단계는 box, class, angle head가 각각 3개 레벨로 출력된다고 가정합니다.
- 실제 실행에는 올바른 Mobilint 런타임 환경과 호환 하드웨어가 필요합니다.
