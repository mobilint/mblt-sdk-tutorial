# 객체 탐지 모델 컴파일

이 튜토리얼은 Mobilint qb 컴파일러를 사용하여 객체 탐지 모델을 컴파일하는 방법에 대한 상세한 안내를 제공합니다.

이 튜토리얼에서는 Ultralytics에서 개발한 COCO 데이터셋으로 사전 훈련된 [YOLO11m](https://docs.ultralytics.com/models/yolo11/) 모델을 사용합니다. 이 모델은 이미지에서 객체를 탐지하는 데 사용할 수 있는 객체 탐지 모델입니다.

## 사전 요구사항

시작하기 전에 다음이 설치되어 있는지 확인하세요:

- qubee SDK 컴파일러 설치 (버전 >= 0.11 필요)

또한 다음 패키지를 설치해야 합니다:

```bash
pip install ultralytics
```

## 개요

컴파일 프로세스는 세 가지 주요 단계로 구성됩니다:

1. **모델 준비**: 모델을 다운로드하고 ONNX 형식으로 내보내기
2. **캘리브레이션 데이터셋 생성**: COCO 데이터셋에서 캘리브레이션 데이터 생성
3. **모델 컴파일**: 캘리브레이션 데이터를 사용하여 모델을 `.mxq` 형식으로 변환

## Step 1: 모델 준비

먼저 모델을 준비해야 합니다. `ultralytics` 라이브러리를 사용하여 사전 훈련된 모델을 다운로드하고 ONNX 형식으로 내보냅니다.

```bash
yolo export model=yolo11m.pt format=onnx # 모델을 ONNX 형식으로 내보내기
```

실행 후, 내보낸 ONNX 모델은 현재 디렉토리에 `yolo11m.onnx`로 저장됩니다.

## Step 2: 캘리브레이션 데이터셋 준비

YOLO11m 모델은 COCO 데이터셋으로 훈련되었으므로 캘리브레이션 데이터셋을 준비해야 합니다.

```bash
wget http://images.cocodataset.org/zips/val2017.zip # 검증 데이터셋 다운로드
unzip val2017.zip # 데이터셋 압축 해제
```

> 참고: [COCO 데이터셋](https://cocodataset.org/#download) 페이지에 따르면 Google Cloud Platform을 통해 데이터셋을 다운로드하는 것이 권장되지만, 현재는 사용할 수 없습니다.

캘리브레이션 데이터셋은 양자화된 모델과 호환되도록 전처리되어야 합니다. 따라서 먼저 원본 모델에서 사용하는 전처리 작업을 조사해야 합니다. 전처리 작업은 [Ultralytics의 GitHub](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py)에 정의되어 있습니다. 다음과 같이 간소화되었지만 동일한 동작을 하는 함수를 작성했습니다:

```python
import numpy as np
import cv2

img_size = [640, 640] 
def preprocess_yolo(img_path: str):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h0, w0 = img.shape[:2]  # original hw
    r = min(img_size[0] / h0, img_size[1] / w0)  # ratio
    new_unpad = int(round(w0 * r)), int(round(h0 * r))
    dh, dw = (
        img_size[0] - new_unpad[1],
        img_size[1] - new_unpad[0],
    )  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if (img.shape[1], img.shape[0]) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # add border
    img = (img / 255).astype(np.float32)

    return img

```

qb 컴파일러의 유틸리티 함수 중 하나인 `make_calib_man`은 사용자 정의 전처리 함수로 캘리브레이션 데이터셋을 생성하는 데 사용할 수 있습니다. `prepare_calib.py` 스크립트는 위에서 정의한 전처리 작업을 사용하여 캘리브레이션 데이터셋을 생성하는 데 이 함수를 사용합니다.

```bash
python3 prepare_calib.py --data_dir {path_to_calibration_dataset} --img_size {image_size} --save_dir {path_to_save_calibration_dataset} --save_name {name_of_calibration_dataset} --max_size {maximum_number_of_calibration_data}
```

**이 스크립트의 동작:**

- COCO 데이터셋을 로드합니다
- 위에서 정의한 전처리 작업을 사용하여 이미지를 전처리합니다
- 전처리된 이미지를 캘리브레이션 데이터로 저장합니다

**매개변수:**

- `--data_dir`: 캘리브레이션 데이터셋 경로
- `--img_size`: 이미지 크기
- `--save_dir`: 캘리브레이션 데이터셋을 저장할 경로
- `--save_name`: 캘리브레이션 데이터셋 이름
- `--max_size`: 최대 캘리브레이션 데이터 수

**출력 위치:**
캘리브레이션 데이터셋은 `--save_dir`로 지정된 디렉토리에 저장됩니다.

예제 명령어는 다음과 같습니다:

```bash
python3 prepare_calib.py --data_dir ./val2017 --img_size 640 --save_dir ./ --save_name yolo11m_cali --max_size 100
```

## Step 3: 모델 컴파일

캘리브레이션 데이터셋과 모델이 준비되면 모델을 컴파일할 수 있습니다.

```bash
python3 model_compile.py --onnx_path {path_to_onnx_model} --calib_data_path {path_to_calibration_dataset} --save_path {path_to_save_model} --quant_percentile {quantization_percentile} --topk_ratio {topk_ratio} --inference_scheme {inference_scheme}
```

**이 스크립트의 동작:**

- ONNX 모델을 로드합니다
- 캘리브레이션 데이터를 로드합니다
- 모델을 `.mxq` 형식으로 컴파일합니다

**매개변수:**

- `--onnx_path`: ONNX 모델 경로
- `--calib_data_path`: 캘리브레이션 데이터 경로
- `--save_path`: MXQ 모델을 저장할 경로
- `--quant_percentile`: 양자화 백분위수
- `--topk_ratio`: Top-k 비율
- `--inference_scheme`: 추론 스키마

**출력 위치:**
컴파일된 모델은 `--save_path`로 지정된 디렉토리에 저장됩니다.

양자화 백분위수와 top-k 비율은 양자화 알고리즘을 실행하는 데 필요한 매개변수입니다.

추론 스키마는 모델의 코어 할당 전략을 지정하는 매개변수입니다. 현재 다음 추론 스키마가 지원됩니다:

- single: 단일 코어 추론
- multi: 다중 코어 추론
- global: 글로벌 추론 (사용 중단됨, global8로 대체됨)
- global4: 4개 코어를 사용한 글로벌 추론
- global8: 8개 코어를 사용한 글로벌 추론

추론 스키마에 대한 자세한 내용은 [Multi-Core Modes](https://docs.mobilint.com/v0.29/en/multicore.html) 문서에서 확인할 수 있습니다.

예제 명령어는 다음과 같습니다:

```bash
python3 model_compile.py --onnx_path ./yolo11m.onnx --calib_data_path ./yolo11m_cali --save_path ./yolo11m.mxq --quant_percentile 0.999 --topk_ratio 0.001 --inference_scheme single
```

위 명령어를 실행하면 현재 디렉토리에 `yolo11m.mxq`로 컴파일된 모델이 저장됩니다.