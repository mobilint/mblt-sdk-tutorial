# Calibration 데이터 가이드

양자화(quantization) 과정에서 모델의 활성값(activation) 분포를 파악하기 위해 **calibration 데이터**가 필요합니다.
이 문서는 calibration 데이터의 목적, 포맷, 그리고 모델별 준비 방법을 설명합니다.

## Calibration 데이터란?

양자화는 모델의 float32 가중치와 활성값을 더 낮은 비트(8bit, 4bit)로 변환하는 과정입니다.
이때 값의 범위를 정확히 알아야 적절한 스케일/오프셋을 결정할 수 있습니다.

Calibration 데이터는 **실제 추론과 유사한 대표 입력 샘플**로,
컴파일러가 이를 모델에 통과시켜 각 레이어의 활성값 분포를 수집합니다.

라벨은 필요 없으며, 실제 추론 시 들어오는 입력 데이터와 유사한 분포의 데이터를 사용하는 것을 권장합니다.

> **주의**: calibration 데이터의 shape은 MBLT의 input shape과 동일해야 합니다.

```text
calibration 데이터 → 모델에 통과 → 활성값 분포 수집 → 양자화 스케일 결정
```

> **PyTorch 양자화와의 차이점**: PyTorch의 `torch.quantization`은 calibration을 선택적으로 사용하지만,
> qbcompiler에서는 NPU 최적화를 위해 calibration이 필수입니다.

---

## 모델별 Calibration 데이터 예시

### 이미지 입력 모델

원본 이미지 파일을 디렉토리에 모아두면 됩니다.
`PreprocessingConfig`를 사용하면 컴파일러가 전처리를 자동 수행하므로, 원본 이미지 그대로 사용할 수 있습니다.

```text
calib_data/
├── image_001.JPEG
├── image_002.JPEG
└── ...
```

- **전달 방식**: 디렉토리 경로를 `calib_data_path`에 전달

> **튜토리얼 참고**: `image_classification/`

---

### 임베딩 입력 모델 (단일 입력)

임베딩 레이어가 NPU에서 실행되지 않는 모델은,
텍스트를 토큰화한 뒤 **임베딩 벡터로 변환한 `.npy` 파일**을 calibration 데이터로 사용합니다.

```text
calibration_data/
├── inputs_embeds_0.npy     # shape: [1, seq_len, embed_dim]
├── inputs_embeds_1.npy
└── ...
```

생성 과정:

```python
# 1. 임베딩 가중치 로드
embedding_weight = torch.load("embedding.pt")
embedding_layer = torch.nn.Embedding.from_pretrained(embedding_weight)

# 2. 텍스트를 토큰화
token_ids = tokenizer(text, return_tensors="pt")["input_ids"]

# 3. 토큰 ID → 임베딩 벡터로 변환
embedded_text = embedding_layer(token_ids)  # [1, seq_len, embed_dim]

# 4. .npy로 저장
np.save("inputs_embeds_0.npy", embedded_text.numpy())
```

> **임베딩을 별도로 추출하는 이유**: NPU는 임베딩 룩업 테이블 연산을 지원하지 않으므로,
> 임베딩 레이어는 CPU에서 실행하고 그 결과를 NPU 모델에 입력합니다.
> 따라서 calibration 데이터도 임베딩 이후 단계의 값이어야 합니다.

> **튜토리얼 참고**: `llm/`, `bert/`

---

### 다중 입력 모델

입력이 여러 텐서로 구성된 모델은 각 입력의 `.npy` 파일 경로를 쌍으로 매핑해야 합니다.
단순 `.npy` 목록으로는 다중 입력을 표현할 수 없으므로 **JSON 형태**를 사용합니다.

```text
calibration_data/
├── sample_0000/
│   ├── decoder_hidden_states.npy
│   └── encoder_hidden_states.npy
├── sample_0001/
│   └── ...
└── calib.json                    # 다중 입력 경로 매핑
```

> **튜토리얼 참고**: `stt/` (encoder/decoder 분리)

---

### 멀티 컴포넌트 모델 (서브모델별 분리)

여러 서브모델로 구성된 모델은 **서브모델별로 calibration 데이터를 따로 준비**합니다.
각 서브모델의 입력 형태가 다르기 때문입니다.

```text
calibration_data/
├── language/
│   ├── sample_000/
│   │   └── inputs_embeds.npy
│   ├── npy_files.txt             # .npy 파일 경로 목록
│   └── metadata.json
└── vision/
    ├── sample_000/
    │   └── images.npy
    ├── npy_files.txt
    └── metadata.json
```

> **튜토리얼 참고**: `vlm/` (vision/language 분리), `stt/` (encoder/decoder 분리)

---

## `npy_files.txt` 형식

VLM, STT 등에서 calibration 데이터 경로를 나열하는 텍스트 파일입니다.

```text
calibration_data/language/sample_000/inputs_embeds.npy
calibration_data/language/sample_001/inputs_embeds.npy
calibration_data/language/sample_002/inputs_embeds.npy
...
```

`mxq_compile()`의 `calib_data_path`에 이 파일 경로를 전달합니다:

```python
mxq_compile(
    ...,
    calib_data_path="calibration_data/language/npy_files.txt",
)
```

---

## Calibration 데이터 품질 팁

| 항목 | 권장 사항 |
|------|----------|
| **샘플 수** | 일반적으로 100~128개. 너무 적으면 분포가 편향되고, 너무 많으면 컴파일 시간 증가 |
| **다양성** | 실제 추론 데이터와 유사한 분포를 가진 다양한 샘플 사용 |
| **시퀀스 길이** | LLM의 경우 최소 512 이상의 시퀀스 길이 권장 |
| **OOM 대응** | calibration 샘플 수를 줄이거나 시퀀스 길이를 줄여 메모리 문제 해결 |

## 다음 문서

- [양자화 설정 가이드](./01_about_quantization_config.KR.md) - calibration 데이터를 활용하는 양자화 config 설명
- [멀티 컴포넌트 모델 가이드](./03_about_multi_component.KR.md) - VLM/STT의 분리된 calibration 구조
