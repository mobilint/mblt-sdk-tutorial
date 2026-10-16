# Calibration 데이터 가이드

양자화(quantization) 과정에서 모델의 활성값(activation) 분포를 파악하기 위해 **calibration 데이터**가 필요합니다.
이 문서는 calibration 데이터의 목적, 포맷, 그리고 모델별 준비 방법을 설명합니다.

## Calibration 데이터란?

양자화는 모델의 float32 가중치와 활성값을 더 낮은 비트(8bit, 4bit)로 변환하는 과정입니다.
이때 값의 범위를 정확히 알아야 적절한 스케일/오프셋을 결정할 수 있습니다.

Calibration 데이터는 **실제 추론과 유사한 대표 입력 샘플**로,
컴파일러가 이를 모델에 통과시켜 각 레이어의 활성값 분포를 수집합니다.

라벨은 필요 없으며, 실제 추론 시 들어오는 입력 데이터와 유사한 분포의 데이터를 사용하는 것을 권장합니다.

> **주의**: calibration 데이터는 quantizer가 MBLT에서 만든 모델의 입력과 같아야 합니다. 아래 `npy_files.json` 절을 참고하십시오.

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
# 1. 모델의 임베딩 레이어 로드
model = AutoModelForCausalLM.from_pretrained(model_id)
embedding_layer = model.get_input_embeddings()

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

VLM 튜토리얼은 서브모델마다 디렉터리 하나를 만듭니다. 구조는 vision 모드에 따라 다릅니다.

dynamic 모드(기본값)입니다. vision 샘플은 입력 3개, language 샘플은 입력 5개입니다:

```text
calibration_data/
├── vision/
│   ├── sample_000/
│   │   ├── float_1_channel_last.npy   # folded pixel values [1, 1, N, 1536]
│   │   ├── pos_embeds.npy             # 위치 임베딩 [1, 1, N, 1024]
│   │   └── cos.npy                    # packed RoPE [1, 1, N, 128]
│   └── npy_files.json                 # 입력 매니페스트, 입력 3개
├── prefill/                           # 중간 저장
│   ├── sample_000/
│   └── npy_files.json                 # 컴파일에 사용하지 않음
├── decode/                            # 중간 저장
│   ├── sample_000/
│   └── npy_files.json                 # 컴파일에 사용하지 않음
└── language/                          # prefill과 decode를 병합. 디코더 컴파일은 이 디렉터리를 사용
    ├── prefill_000/
    │   ├── inputs_embeds.npy          # [1, 1, T, 2048]
    │   ├── deepstack_0.npy            # [1, 1, T, 2048]
    │   ├── deepstack_1.npy
    │   ├── deepstack_2.npy
    │   └── cos.npy                    # packed RoPE [1, 1, T, 256]
    ├── decode_000/
    │   └── ...                        # 같은 파일 5개
    └── npy_files.json                 # 입력 매니페스트, 입력 5개
```

static 모드(`--static`)입니다. vision 샘플은 입력 1개이므로 vision 목록은 `.txt` 파일입니다:

```text
calibration_data_static/
├── vision/
│   ├── sample_000/
│   │   └── images.npy                 # folded pixel values [1, 256, 1536]
│   └── npy_files.txt                  # 경로 목록, 입력 1개
├── prefill/                           # 중간 저장
│   ├── sample_000/
│   └── npy_files.json                 # 컴파일에 사용하지 않음
├── decode/                            # 중간 저장
│   ├── sample_000/
│   └── npy_files.json                 # 컴파일에 사용하지 않음
└── language/                          # prefill과 decode를 병합. 디코더 컴파일은 이 디렉터리를 사용
    ├── prefill_000/
    │   ├── inputs_embeds.npy
    │   ├── deepstack_0.npy
    │   ├── deepstack_1.npy
    │   └── deepstack_2.npy            # cos.npy 없음: dynamic_rope=False이므로 quantizer가 RoPE 입력을 추가하지 않음
    ├── decode_000/
    │   └── ...
    └── npy_files.json                 # 입력 매니페스트, 입력 4개
```

> **튜토리얼 참고**: `vlm/` (vision/language 분리), `stt/` (encoder/decoder 분리)

---

## `npy_files.txt` 형식

calibration 데이터 경로를 나열하는 텍스트 파일입니다.
서브모델의 입력이 하나일 때 사용합니다. 예를 들어 static VLM vision 모드(`npy_files.txt`)와 STT 인코더(`whisper_encoder_cali.txt`)가 해당합니다.
생성된 파일은 절대 경로를 사용합니다.

```text
calibration_data_static/vision/sample_000/images.npy
calibration_data_static/vision/sample_001/images.npy
calibration_data_static/vision/sample_002/images.npy
...
```

`mxq_compile()`의 `calib_data_path`에 이 파일 경로를 전달합니다:

```python
mxq_compile(
    ...,
    target_device="aries-rb",
    calib_data_path="calibration_data_static/vision/npy_files.txt",
)
```

---

## `npy_files.json` 형식

입력 이름, 입력 shape, 샘플별 경로 목록을 담은 JSON 매니페스트입니다.
서브모델의 입력이 여러 개일 때 사용합니다. 예를 들어 Qwen3-VL vision 및 language 파트가 해당합니다.
VLM 튜토리얼은 기본 dynamic 모드에서 이 매니페스트를 작성합니다.
아래는 dynamic vision 매니페스트를 샘플 하나로 줄인 것입니다:

```json
{
    "info": {
        "input names": ["float_1_channel_last", "pos_embeds/reshape", "cos"],
        "input shapes": [[1, 1, -1, 1536], [1, 1, -1, 1024], [1, 1, -1, 128]]
    },
    "calib paths": [
        [
            "calibration_data/vision/sample_000/float_1_channel_last.npy",
            "calibration_data/vision/sample_000/pos_embeds.npy",
            "calibration_data/vision/sample_000/cos.npy"
        ]
    ]
}
```

생성된 파일은 절대 경로를 사용합니다.
`-1` 축은 dynamic 축을 뜻합니다.
입력 수와 shape는 quantizer가 MBLT에서 만든 모델의 입력과 같아야 합니다.
이 입력은 MBLT graph 입력과 다를 수 있습니다. 예를 들면 다음과 같습니다:

- Qwen3-VL vision MBLT는 폭 64의 `cos`, `sin` 입력을 따로 가집니다. quantizer는 이 둘을 폭 128의 RoPE 입력 하나로 합치므로 매니페스트에는 폭 128의 `cos` 항목 하나만 있습니다.
- Qwen3-VL 디코더 MBLT는 입력이 4개이고 RoPE 입력이 없습니다. dynamic 모드에서는 컴파일 설정이 `dynamic_rope=True`이고, quantizer가 폭 256의 RoPE 입력을 추가합니다. 그래서 language 매니페스트는 입력 5개입니다. 폭 2048 입력 4개(`inputs_embeds/reshape`, `deepstack_visual_embeds/reshape/slice`, `deepstack_visual_embeds/reshape/slice_0`, `deepstack_visual_embeds/reshape/slice_1`)와 폭 256의 `cos`입니다. static 모드에서는 `dynamic_rope=False`이므로 매니페스트에 입력 4개만 있습니다.

shape가 같은 입력이 여러 개이면 quantizer는 입력 이름으로 파일을 매칭하고, 모델 입력 이름이 아닌 이름은 거부합니다.
shape가 유일하면 quantizer는 이름을 검사하지 않습니다.
어떤 경우에도 이름은 모델 입력 이름과 같게 두십시오.

---

## Calibration 데이터 품질 팁

| 항목 | 권장 사항 |
| ------ | ---------- |
| **샘플 수** | 일반적으로 100~128개. 너무 적으면 분포가 편향되고, 너무 많으면 컴파일 시간 증가 |
| **다양성** | 실제 추론 데이터와 유사한 분포를 가진 다양한 샘플 사용 |
| **시퀀스 길이** | LLM의 경우 최소 512 이상의 시퀀스 길이 권장 |
| **OOM 대응** | calibration 샘플 수를 줄이거나 시퀀스 길이를 줄여 메모리 문제 해결. qbcompiler 1.4부터 weight quantization의 GPU 메모리는 자동으로 예산이 정해지며, `ResourceManagementConfig(gpu_memory_budget_mb=...)`(MiB, `-1` 자동, `0` 제한 없음)로 바꿀 수 있음 |

## 다음 문서

- [양자화 설정 가이드](./01_about_quantization_config.KR.md) - calibration 데이터를 활용하는 양자화 config 설명
- [멀티 컴포넌트 모델 가이드](./03_about_multi_component.KR.md) - VLM/STT의 분리된 calibration 구조
