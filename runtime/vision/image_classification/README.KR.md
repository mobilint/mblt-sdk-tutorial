# 이미지 분류 모델 추론 (Image Classification Model Inference)

이 튜토리얼은 Mobilint qbruntime을 사용하여 컴파일된 이미지 분류 모델로 추론을 실행하는 방법에 대한 단계별 지침을 제공합니다.

이 가이드는 [mblt-sdk-tutorial/compilation/vision/image_classification/README.md](file:///workspace/mblt-sdk-tutorial/compilation/vision/image_classification/README.md)에서 이어지는 내용입니다. 모델 컴파일을 성공적으로 마쳤으며 다음 파일이 준비되어 있다고 가정합니다:

- `./resnet50.mxq` - 컴파일된 모델 파일

## 사전 요구 사항 (Prerequisites)

추론을 실행하기 전에 다음 구성 요소가 설치되어 있고 준비되었는지 확인하십시오:

- `qbruntime` 라이브러리 (NPU 가속기 액세스용)
- 컴파일된 `.mxq` 모델 파일
- Python 패키지: `PIL`, `numpy`, `torch`

## 개요 (Overview)

추론 로직은 `inference_mxq.py` 스크립트에 구현되어 있습니다. 이 스크립트는 다음 워크플로우를 보여줍니다:

1.  **모델 로드**: `qbruntime`을 통해 컴파일된 `.mxq` 모델을 로드합니다.
2.  **전처리**: 입력 이미지를 준비합니다 (리사이즈, 중앙 크롭).
3.  **추론**: NPU 가속기에서 모델을 실행합니다.
4.  **결과 출력**: 상위 5개 분류 결과와 해당 확률을 출력합니다.

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
def preprocess_resnet50(img_path: str) -> np.ndarray:
    """ResNet-50을 위한 이미지 전처리"""
    img = Image.open(img_path).convert("RGB")
    resize_size = 256
    crop_size = (224, 224)
    out = F.pil_to_tensor(img)
    out = F.resize(out, size=resize_size, interpolation=InterpolationMode.BILINEAR)
    out = F.center_crop(out, output_size=crop_size)
    out = np.transpose(out.numpy(), axes=[1, 2, 0])
    out = out.astype(np.uint8)
    return out

image = preprocess_resnet50(args.image_path)
```

마지막으로, 추론을 실행하고 예측 확률을 얻습니다.

```python
output = mxq_model.infer(image)

output = output[0].reshape(-1).astype(np.float32)
output = np.exp(output) / np.sum(np.exp(output)) # softmax
```

예제 추론 스크립트를 실행하려면 다음 명령어를 사용하십시오:

```bash
python inference_mxq.py --mxq_path ../../../compilation/vision/image_classification/resnet50.mxq --image_path ../rc/volcano.jpg
```

### 스크립트 세부 설명

- **모델 실행 (Model Execution)**: `.mxq` 파일을 NPU에 로드합니다.
- **전처리 (Preprocessing)**: 이미지를 256px로 리사이즈하고, 224x224 크기로 중앙 크롭(center crop)을 수행하며, `UInt8` 형식의 데이터를 유지합니다 (정규화는 NPU에서 수행됨).
- **추론 (Inference)**: NPU에서 모델의 순전파(forward pass)를 실행합니다.
- **결과 출력 (Result Display)**: 상위 5개 예측 클래스와 해당 신뢰도 점수를 출력합니다.

### 예상 출력 (Expected Output)

스크립트는 이미지 형태와 상위 5개 예측 클래스 및 신뢰도 점수를 출력합니다.
