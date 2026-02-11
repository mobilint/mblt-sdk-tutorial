# 인스턴스 세그멘테이션 모델 추론 (Instance Segmentation Model Inference)

이 튜토리얼은 Mobilint qbruntime을 사용하여 컴파일된 인스턴스 세그멘테이션 모델로 추론을 실행하는 방법에 대한 단계별 지침을 제공합니다.

이 가이드는 [mblt-sdk-tutorial/compilation/vision/instance_segmentation/README.md](file:///workspace/mblt-sdk-tutorial/compilation/vision/instance_segmentation/README.md)에서 이어지는 내용입니다. 모델 컴파일을 성공적으로 마쳤으며 다음 파일이 준비되어 있다고 가정합니다:

- `./yolo11m-seg.mxq` - 컴파일된 모델 파일

## 사전 요구 사항 (Prerequisites)

추론을 실행하기 전에 다음 구성 요소가 설치되어 있고 준비되었는지 확인하십시오:

- `qbruntime` 라이브러리 (NPU 가속기 액세스용)
- 컴파일된 `.mxq` 모델 파일
- Python 패키지: `opencv-python`, `numpy`, `torch`

## 개요 (Overview)

추론 로직은 `inference_mxq.py` 스크립트에 구현되어 있습니다. 이 스크립트는 다음 워크플로우를 보여줍니다:

1.  **모델 로드**: `qbruntime`을 통해 컴파일된 `.mxq` 모델을 로드합니다.
2.  **전처리**: 입력 이미지를 준비합니다 (예: 레터박싱을 포함한 리사이즈).
3.  **추론**: NPU 가속기에서 모델을 실행합니다.
4.  **후처리**: 모델 출력을 처리합니다 (바운딩 박스 좌표 및 세그멘테이션 마스크 디코딩, Non-Maximum Suppression 적용).
5.  **시각화**: 바운딩 박스, 라벨, 세그멘테이션 마스크를 원본 이미지에 그립니다.

후처리에 필요한 연산을 더 잘 이해하려면 컴파일 과정에서 생성된 `.mblt` 파일을 [Mobilint Netron](https://netron.mobilint.com/)을 사용하여 검사해 볼 수 있습니다.

## 추론 실행 (Running Inference)

`inference_mxq.py` 스크립트는 세부적인 단계로 추론을 수행합니다.

먼저, NPU 가속기와 모델 설정을 초기화합니다.

```python
acc = qbruntime.Accelerator(0)
mc = qbruntime.ModelConfig()
mc.set_single_core_mode(1)
mxq_model = qbruntime.Model(args.mxq_path, mc)
mxq_model.launch(acc)
```

다음으로, 입력 이미지를 로드하고 전처리합니다. 컴파일 과정에서 정규화 연산이 MXQ 모델에 융합(fused)되었으므로, 입력 이미지는 `UInt8` 형식을 유지해야 합니다.

```python
def preprocess_yolo(img_path: str, img_size=(640, 640)):
    # 참고: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py#L1535
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]  # 원본 높이 및 너비
    r = min(img_size[0] / h0, img_size[1] / w0)  # 스케일 비율
    new_unpad = int(round(w0 * r)), int(round(h0 * r))

    if (w0, h0) != new_unpad:  # 리사이즈
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    dh, dw = img_size[0] - new_unpad[1], img_size[1] - new_unpad[0]  # 너비 및 높이 패딩
    dw /= 2  # 양쪽 패딩 분할
    dh /= 2  # 이미지 중앙 정렬
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # 테두리 패딩 추가

    return img
```

마지막으로, 전처리된 입력으로 모델을 실행하고 후처리를 적용하여 결과를 해석합니다.

예제 추론 스크립트를 실행하려면 다음 명령어를 사용하십시오:

```bash
python inference_mxq.py --model-path ../../../compilation/vision/instance_segmentation/yolo11m-seg.mxq --image-path ../rc/cr7.jpg --output-path tmp/cr7.jpg --conf-thres 0.25 --iou_thres 0.45
```

### 스크립트 세부 설명

- **모델 실행 (Model Execution)**: `.mxq` 파일을 NPU에 로드합니다.
- **전처리 (Preprocessing)**: 종횡비를 유지하며 이미지를 640x640 크기로 리사이즈하고(레터박싱), 회색 테두리로 패딩을 추가하며, 데이터를 적절한 형식으로 유지합니다.
- **추론 (Inference)**: NPU에서 모델의 순전파(forward pass)를 실행합니다.
- **후처리 (Postprocessing)**: 원시 출력(raw output)을 바운딩 박스와 세그멘테이션 마스크로 디코딩하고, 신뢰도 점수로 필터링하며, Non-Maximum Suppression (NMS)을 적용합니다.
- **시각화 (Visualization)**: 감지된 바운딩 박스, 클래스 라벨, 세그멘테이션 마스크를 출력 이미지 위에 겹쳐 그립니다.

### 파라미터 (Parameters)

- `--model-path`: 컴파일된 `.mxq` 모델 파일의 경로입니다.
- `--image-path`: 입력 이미지 파일의 경로입니다.
- `--output-path`: (선택 사항) 출력 이미지가 저장될 경로입니다. 지정하지 않으면 현재 디렉토리에 `output.jpg`로 저장됩니다.
- `--conf-thres`: 감지 결과를 필터링하기 위한 신뢰도 임계값입니다 (기본값: `0.25`).
- `--iou-thres`: NMS를 위한 IoU (Intersection over Union) 임계값입니다 (기본값: `0.45`).

### 예상 출력 (Expected Output)

스크립트는 감지 결과(라벨 및 신뢰도 점수)를 콘솔에 출력하고, 바운딩 박스와 마스크가 그려진 이미지를 `tmp/cr7.jpg`에 저장합니다.
