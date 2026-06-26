# 회전 바운딩 박스 런타임

이 튜토리얼은 Mobilint `qbruntime`를 사용해 컴파일된 `YOLO11m-obb` MXQ 모델을 실행하는 방법을 설명합니다.

이 가이드를 진행하기 전에 [../../../compilation/oriented_bounding_boxes/README.KR.md](../../../compilation/oriented_bounding_boxes/README.KR.md)의 컴파일 단계를 먼저 완료하세요. 이 런타임 예제는 컴파일 결과물이 `../../../compilation/oriented_bounding_boxes/yolo11m-obb.mxq`에 있다고 가정합니다.

## 사전 준비

다음 패키지가 준비되어 있어야 합니다:

- `qbruntime`
- `opencv-python`
- `numpy`
- `torch`

SDK 환경에 포함되지 않은 Python 패키지는 다음과 같이 설치할 수 있습니다:

```bash
pip install opencv-python numpy torch
```

## 개요

이 디렉토리의 런타임 파이프라인은 다음 다섯 단계로 구성됩니다:

1. `qbruntime`로 `yolo11m-obb.mxq`를 로드합니다.
2. 컴파일 단계와 동일한 `1024x1024` letterbox 전처리를 적용합니다.
3. Mobilint 런타임에서 MXQ 추론을 실행합니다.
4. DOTA용 회전 바운딩 박스를 디코드하고 rotated NMS를 적용합니다.
5. 클래스 이름과 신뢰도를 포함한 회전 폴리곤을 원본 이미지에 렌더링합니다.

`/255` 정규화는 컴파일 시 `Uint8InputConfig`로 이미 MXQ 모델에 융합되어 있으므로, 이 런타임 예제는 입력을 `uint8` 그대로 전달합니다.

## 이 튜토리얼의 파일

- `inference_mxq.py`: MXQ 추론을 실행하고 렌더링된 결과 이미지를 저장합니다.
- `postprocess.py`: OBB 출력을 `cx, cy, w, h, conf, cls, angle` 형식의 행 단위 텐서로 디코드합니다.
- `visualize.py`: 회전 박스를 원본 이미지 좌표계로 복원하고 폴리곤을 그립니다.
- `dota.py`: DOTAv1 클래스 이름과 고정 색상 팔레트를 정의합니다.
- `utils.py`: 최소한의 DFL, 회전 박스 디코드, 좌표 복원, rotated NMS 유틸리티를 제공합니다.

## 예제 실행

기본 모델 경로, 샘플 이미지, 출력 경로를 그대로 사용하려면 다음과 같이 실행합니다:

```bash
python inference_mxq.py
```

이 명령은 다음 기본값을 사용합니다:

- 모델: `../../../compilation/oriented_bounding_boxes/yolo11m-obb.mxq`
- 입력 이미지: `../rc/airport.jpg`
- 출력 이미지: `./tmp/airport_demo.jpg`

임곗값이나 파일 경로를 바꾸려면 다음과 같이 실행할 수 있습니다:

```bash
python inference_mxq.py --conf-thres 0.3 --iou-thres 0.5 --output-path ./tmp/airport_custom.jpg
```

## 참고

- 이 튜토리얼은 `YOLO11m-obb`의 출력 레이아웃만을 대상으로 합니다.
- `visualize.py`가 소비하는 postprocess 출력 형식은 `cx, cy, w, h, conf, cls, angle`입니다.
- 실제 실행에는 올바른 Mobilint 런타임 환경과 호환 하드웨어가 필요합니다. 해당 환경이 없으면 정적 검증까지만 수행할 수 있습니다.
