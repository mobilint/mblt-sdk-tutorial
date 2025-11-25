# 이미지 분류 모델 추론

이 튜토리얼은 Mobilint qb 런타임을 사용하여 컴파일된 이미지 분류 모델로 추론을 실행하는 방법에 대한 상세한 지침을 제공합니다.

이 가이드는 `mblt-sdk-tutorial/compilation/vision/image_classification/README.md`에서 이어집니다. 모델을 성공적으로 컴파일했으며 다음 파일들이 준비되어 있다고 가정합니다:

- `./resnet50.mxq` - 컴파일된 모델 파일

## 사전 요구사항

추론을 실행하기 전에 다음이 준비되어 있는지 확인하세요:

- maccel 런타임 라이브러리 (NPU 가속기 접근 제공)
- 컴파일된 `.mxq` 모델 파일
- Python 패키지: `PIL`, `numpy`, `torch`, `torchvision`

## 개요

추론 프로세스는 `inference_mxq.py` 스크립트에 구현되어 있습니다. 이 스크립트는 다음을 수행하는 방법을 보여줍니다:

- maccel 런타임을 사용하여 컴파일된 `.mxq` 모델 로드
- 입력 이미지 전처리 (크기 조정, 자르기, 정규화)
- NPU 가속기에서 추론 실행
- 상위 5개 분류 결과와 확률 출력

## 추론 실행

예제 추론 스크립트를 실행하려면:

```bash
python inference_mxq.py --mxq_path ../../../compilation/vision/image_classification/resnet50.mxq --image_path ../rc/volcano.jpg
```

**이 스크립트의 동작:**

- 컴파일된 `.mxq` 모델을 NPU 가속기에 로드합니다
- 샘플 이미지를 로드하고 전처리합니다 (ResNet-50 전처리: 256px로 크기 조정, 224x224 중앙 자르기, 정규화)
- NPU 가속기에서 추론을 실행합니다
- 상위 5개 분류 결과와 확률을 출력합니다

**예상 출력:**

스크립트는 이미지 형태와 상위 5개 예측 클래스 및 신뢰도 점수를 표시합니다.
