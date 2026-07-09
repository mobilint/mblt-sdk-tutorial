# 이미지 분류 런타임

이 튜토리얼은 Mobilint `qbruntime`를 사용해 컴파일된 이미지 분류 MXQ 모델을 실행하는 방법을 설명합니다.

시작하기 전에 [../../../compilation/image_classification/README.KR.md](../../../compilation/image_classification/README.KR.md)의 컴파일 과정을 먼저 완료하세요. 이 디렉토리의 런타임 예제는 컴파일된 모델이 `../../../compilation/image_classification/resnet50.mxq`에 있다고 가정합니다.

## 사전 준비

다음 구성 요소가 준비되어 있어야 합니다.

- Mobilint `qbruntime`
- 컴파일된 `.mxq` 모델 파일
- Python 패키지: `Pillow`, `numpy`, `torch`, `torchvision`

Python 패키지가 아직 설치되어 있지 않다면 다음 명령으로 설치할 수 있습니다.

```bash
pip install pillow numpy torch torchvision
```

## 개요

런타임 흐름은 `inference_mxq.py`에 구현되어 있으며 다음 단계를 따릅니다.

1. `qbruntime`로 컴파일된 ResNet-50 MXQ 모델을 로드합니다.
2. 입력 이미지를 읽고 resize 및 center crop 전처리를 적용합니다.
3. Mobilint NPU에서 추론을 실행합니다.
4. softmax를 적용해 로그잇을 확률로 변환합니다.
5. 상위 5개 ImageNet 예측 결과를 출력합니다.

컴파일된 MXQ 모델에는 보통 정규화가 포함되어 있으므로, 이 예제는 런타임 입력을 `uint8` 형식으로 유지합니다.

## 이 튜토리얼의 파일

- `inference_mxq.py`: 전체 추론 흐름을 실행하고 상위 5개 예측 결과를 출력합니다.
- `imagenet.py`: 클래스 인덱스를 ImageNet 라벨로 매핑합니다.

## 스크립트 동작 방식

스크립트는 먼저 accelerator를 초기화하고 컴파일된 모델을 실행합니다.

```python
acc = qbruntime.Accelerator(0)
mc = qbruntime.ModelConfig()
mc.set_single_core_mode(None, [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)])
mxq_model = qbruntime.Model(args.mxq_path, mc)
mxq_model.launch(acc)
```

그 다음 이미지를 읽고 `256`으로 리사이즈한 뒤 `224x224` center crop을 적용하고, 결과를 HWC 형식의 NumPy 배열로 변환합니다.

```python
def preprocess_resnet50(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    resize_size = [256]
    crop_size = [224, 224]
    out = F.pil_to_tensor(img)
    out = F.resize(out, size=resize_size, interpolation=InterpolationMode.BILINEAR)
    out = F.center_crop(out, output_size=crop_size)
    out = np.transpose(out.numpy(), axes=[1, 2, 0])
    # Option 1: normalization is fused into the model/runtime.
    out = out.astype(np.uint8)

    # Option 2: normalization is not fused.
    # out = out.astype(np.float32) / 255.0
    # out = (out - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
    #       np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return out
```

추론 후에는 출력 로그잇을 reshape한 뒤 softmax를 적용하고, 상위 5개 ImageNet 클래스와 확률을 출력합니다.

## 예제 실행

기본 샘플 경로를 그대로 사용하려면 다음 명령을 실행하세요.

```bash
python inference_mxq.py
```

이 명령은 다음 기본값을 사용합니다.

- 모델: `../../../compilation/image_classification/resnet50.mxq`
- 입력 이미지: `../rc/volcano.jpg`

경로를 명시적으로 지정하려면 다음과 같이 실행하세요.

```bash
python inference_mxq.py --mxq-path ../../../compilation/image_classification/resnet50.mxq --image-path ../rc/volcano.jpg
```

## 파라미터

- `--mxq-path`: 컴파일된 `.mxq` 모델 경로
- `--image-path`: 입력 이미지 경로

## 예상 출력

스크립트는 전처리된 이미지 shape와 상위 5개 ImageNet 예측 결과 및 확률을 출력합니다.
