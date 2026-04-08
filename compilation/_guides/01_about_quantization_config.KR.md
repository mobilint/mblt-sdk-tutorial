# 컴파일 Config 가이드

`mxq_compile()`에 전달하는 config 객체들을 소개합니다.

양자화 config는 파라미터 조합이 다양하며, 모델 아키텍처와 태스크에 따라 최적의 값이 다릅니다.
동일한 설정이라도 모델에 따라 정확도와 성능에 미치는 영향이 달라질 수 있으므로,
각 튜토리얼 디렉토리(`image_classification/`, `llm/`, `vlm/` 등)에서 유사한 모델이나 태스크의
컴파일 스크립트를 참고하여 베이스라인 config로 활용하는 것을 권장합니다.

---

## Config 종류 overview

| Config | 역할 | 사용 대상 |
|--------|------|----------|
| `CalibrationConfig` | quantization range 결정 방식 (per-channel, percentile 등) | quantization이 필요한 모든 모델 |
| `BitConfig` | transformer component별 quantization bit 수 (8bit/4bit) | transformer 기반 모델 (LLM 등) |
| `LlmConfig` | sequence length, KV cache, NPU core 할당 | transformer decoder 구조 (autoregressive + KV cache) |
| `EquivalentTransformationConfig` | SpinQuant 등 quantization 오차를 줄이는 고급 수학적 변환 | 4bit quantization에서 사용 권장 |
| `SearchWeightScaleConfig` | layer별 weight scale 학습으로 quantization 정확도 보정 | 4bit quantization에서 사용 권장 |
| `PreprocessingConfig` | calibration 진행 시 image 전처리를 compiler가 자동 수행 | image 입력 모델 (Image Classification 등) |

---

## CalibrationConfig

양자화 시 활성값의 범위를 결정하는 방법을 설정합니다.
per-channel/per-tensor 방식과 percentile 기반 클리핑 등을 제어합니다.

> **PyTorch 양자화와의 차이점**: PyTorch의 기본 양자화는 min-max 방식을 사용하지만,
> qbcompiler는 percentile 기반 클리핑을 사용하여 이상치(outlier)의 영향을 줄입니다.

```python
from qbcompiler import CalibrationConfig

calibration_config = CalibrationConfig(
    # Weight 양자화 + Activation 양자화 방식의 조합을 선택.
    #   0: WChALayer     — Weight: per-channel, Activation: per-layer.
    #                       가중치는 채널별 스케일, 활성값은 레이어 단위 스케일 사용.
    #   1: WChAMulti     — Weight: per-channel, Activation: multi-layer.
    #                       가중치는 채널별, 활성값은 여러 레이어의 통계를 종합하여
    #                       스케일을 결정. 단일 레이어 대비 안정적인 양자화.
    #   2: WChALayerZeropoint  — 0번과 동일 + activation에 zeropoint 적용.
    #   3: WChAMultiZeropoint  — 1번과 동일 + activation에 zeropoint 적용.
    #                            비대칭 분포의 활성값에 유리.
    method=1,

    # 출력(output) 양자화 방식 설정.
    #   0: Layer   — 레이어 전체 출력에 하나의 스케일 적용.
    #   1: Ch      — 출력 채널마다 개별 스케일 적용.
    #   2: Sigmoid — Sigmoid 기반 양자화.
    output=0,

    # 양자화 범위를 결정하는 클리핑 모드.
    #   0: Max           — 활성값의 최대/최소값을 그대로 양자화 범위로 사용.
    #                       단순하지만 이상치(outlier)에 민감.
    #   1: MaxPercentile — 백분위수 기준으로 클리핑 범위 결정.
    #                       이상치를 제외하여 양자화 오차를 줄임.
    #   2: Histogram     — 활성값 히스토그램 기반으로 최적 클리핑 범위를 탐색.
    #                       KL-divergence 등을 활용하여 정보 손실을 최소화.
    mode=1,

    # MaxPercentile 세부 설정
    max_percentile=CalibrationConfig.MaxPercentile(
        # 활성값의 몇 번째 백분위수에서 클리핑할지 지정.
        # 0.9999 = 99.99번째 백분위수. 상위 0.01% 이상치를 클리핑하여
        # 양자화 범위가 이상치에 의해 과도하게 확장되는 것을 방지.
        percentile=0.9999,

        # 백분위수 계산 시 상위 몇 %의 값을 별도로 보존할지 지정.
        # 0.01 = 상위 1%. 중요한 큰 활성값이 클리핑으로 손실되지 않도록
        # 별도 관리하여 정확도를 유지.
        topk_ratio=0.01,
    ),
)
```

**실제 사용 예시**:
- `image_classification/model_compile.py`
- `llm/generate_mxq.py`
- `bert/compile_mxq.py`

---

## BitConfig

트랜스포머 레이어의 각 구성 요소별 양자화 비트 수를 지정합니다.
8bit과 4bit을 선택하거나, value만 8bit로 유지하는 등의 혼합 설정이 가능합니다.

```python
from qbcompiler import BitConfig

bit_config = BitConfig(
    transformer=BitConfig.Transformer(
        weight=BitConfig.Transformer.Weight(
            # Attention 레이어의 각 projection 가중치 비트 수를 개별 지정.
            # 모든 컴포넌트를 동일한 비트로 설정하거나,
            # 일부 레이어만 8bit 값을 유지하는 혼합 설정(예: w4v8)도 가능.

            query=8,    # Q projection — 입력을 query 벡터로 변환하는 가중치
            key=8,      # K projection — 입력을 key 벡터로 변환하는 가중치
            value=8,    # V projection — 입력을 value 벡터로 변환하는 가중치
            output=8,   # Output projection — attention 출력을 다음 레이어로 전달하는 가중치
            ffn=8,      # Feed-Forward Network — 트랜스포머의 FFN 블록 가중치
            head=8,     # Attention head — multi-head attention의 head 가중치
        ),
    )
)
```

**실제 사용 예시**:
- `llm/generate_mxq.py` - 8bit
- `llm/generate_mxq_4bit.py` - 4bit (w4, w4v8)

---

## LlmConfig

LLM 컴파일을 위한 시퀀스 길이, KV 캐시, NPU 코어 할당을 설정합니다.
LLM 전용 config이지만, STT decoder처럼 autoregressive 구조를 갖는 서브모델에서도
동일한 KV 캐시 관리가 필요하므로 함께 사용됩니다.

```python
from qbcompiler import LlmConfig

llm_config = LlmConfig(
    # LLM 전용 설정 활성화 여부
    apply=True,

    attributes=LlmConfig.Attributes(
        # 한 번에 모델에 입력할 수 있는 최대 토큰 수.
        # prefill 단계에서 처리하는 프롬프트 길이 상한.
        max_data_length=4096,

        # 생성을 포함한 전체 시퀀스의 최대 길이.
        # prefill 입력 + 생성 토큰 수의 합이 이 값을 넘을 수 없음.
        max_sequence_length=4096,

        # KV 캐시에 저장할 수 있는 최대 토큰 수.
        # autoregressive 생성 시 이전 토큰의 key/value를 캐시하는 버퍼 크기.
        # 이 값이 클수록 긴 문맥을 유지할 수 있으나 메모리 사용량 증가.
        max_cache_length=4096,

        # NPU 코어 하나가 한 번에 처리하는 데이터 버퍼 크기.
        # NPU 내부 메모리 할당에 영향을 미치는 하드웨어 레벨 파라미터.
        max_core_data_length=128,

        calibration=LlmConfig.Attributes.Calibration(
            # True: calibration 시 전체 시퀀스 길이를 사용하여 활성값 분포를 수집.
            # False: 일부 시퀀스만 사용. True가 정확도에 유리하나 메모리를 더 사용.
            use_full_seq_length=True,
        ),
    ),
)
```

**실제 사용 예시**:
- `llm/generate_mxq.py` - LLM 컴파일 시 시퀀스/캐시 길이 설정
- `stt/compile_decoder.py` - Whisper decoder (autoregressive 구조이므로 LlmConfig 필요)

---

## EquivalentTransformationConfig

양자화 정확도 손실을 줄이기 위한 고급 수학적 변환(SpinQuant 등)을 설정합니다.
컴파일 시 회전 행렬(spinWeight)을 생성합니다.
양자화 오차가 큰 4bit 양자화에서 사용을 권장합니다.

> **주의**: SpinR1을 사용하는 경우, 컴파일 후 임베딩 가중치에 R1 회전을 수동으로 적용하는
> 추가 작업이 필요합니다. 자세한 내용은 아래 [SpinQuant (R1/R2) 상세 설명](#spinquant-r1r2-상세-설명)을 참조하세요.

```python
from qbcompiler import EquivalentTransformationConfig

et_config = EquivalentTransformationConfig(
    # Normalization-Convolution 등가 변환.
    # LayerNorm/RMSNorm의 스케일을 후속 linear 레이어에 흡수시켜
    # 양자화 친화적인 가중치 분포로 변환.
    norm_conv=EquivalentTransformationConfig.NormConv(apply=True),

    # Query-Key 등가 변환.
    # Q와 K의 가중치를 회전하여 attention score의 양자화 오차를 줄임.
    qk=EquivalentTransformationConfig.Qk(apply=True),

    # Up-Down 등가 변환.
    # FFN의 up/down projection 가중치를 회전하여 양자화 오차를 줄임.
    ud=EquivalentTransformationConfig.Ud(apply=True),

    # Value-Output 등가 변환.
    # V와 output projection 가중치를 회전하여 attention 출력의 양자화 오차를 줄임.
    vo=EquivalentTransformationConfig.Vo(apply=True),

    # SpinQuant R1 — 모델 전체에 적용하는 전역 회전 행렬.
    # 가중치 공간을 회전시켜 양자화에 유리한 분포로 변환.
    # 컴파일 시 spinWeight/{model}/R1/global_rotation.pth 파일 생성.
    # 4bit 양자화 시 임베딩 가중치에 이 회전을 미리 적용해야 함.
    spin_r1=EquivalentTransformationConfig.SpinR1(apply=True),

    # SpinQuant R2 — 각 트랜스포머 레이어에 개별 적용하는 회전 행렬.
    # 레이어별 가중치 분포 차이를 보정하여 R1보다 세밀한 최적화 수행.
    # 컴파일 시 spinWeight/{model}/R2/ 디렉토리에 레이어별 파일 생성.
    spin_r2=EquivalentTransformationConfig.SpinR2(apply=True),

    # QK Rotation — RoPE 등 positional encoding과의 호환성을 유지하면서
    # Q/K 가중치를 회전하는 변환.
    qk_rotation=EquivalentTransformationConfig.QkRotation(apply=True),

    # FFN Multi-LUT — FFN 가중치를 여러 룩업 테이블로 분해하여
    # 4bit 양자화에서의 표현력을 높임.
    feed_forward_multi_lut=EquivalentTransformationConfig.FeedForwardMultiLut(apply=True),

    # FFN 최적화 — FFN 블록의 연산 구조를 NPU에 맞게 재배치.
    optimize_ffn=EquivalentTransformationConfig.OptimizeFfn(apply=True),
)
```

**실제 사용 예시**:
- `llm/generate_mxq_4bit.py` - LLM 4bit SpinQuant 적용
- `vlm/mxq_compile_language.py` - VLM language 모델의 등가 변환
- `vlm/mxq_compile_vision.py` - VLM vision encoder에서 R1 회전 행렬 참조 (`HeadOutChRotation`)

### SpinQuant (R1/R2) 상세 설명

SpinQuant는 4bit 양자화에서 정확도 손실을 줄이기 위해 가중치 공간을 회전하는 기법입니다.
([SpinQuant: LLM Quantization with Learned Rotations](https://arxiv.org/abs/2405.16406) 논문 참고)

컴파일 시 `spinWeight/` 디렉토리에 회전 행렬 파일이 생성됩니다.

```text
spinWeight/{model_name}/
├── R1/
│   └── global_rotation.pth     # 전역 회전 행렬 (모델 전체에 1개)
└── R2/
    └── layer_*.pth             # 레이어별 회전 행렬 (레이어마다 1개)
```

**R1 (전역 회전)** 은 모델의 전체 가중치 공간을 하나의 회전 행렬로 변환합니다.
이 회전은 컴파일된 MXQ 모델 내부에 이미 반영되지만,
**임베딩 레이어는 MXQ에 포함되지 않고 CPU에서 실행**되므로
추론 전에 임베딩 가중치에 동일한 R1 회전을 수동으로 적용해야 합니다.

- SpinQuant(R1)를 사용하지 않는 경우: 임베딩 회전 불필요
- SpinQuant(R1)를 사용하는 경우: 임베딩에 R1 회전 필수

LLM 임베딩 회전 예시 (`llm/get_rotation_emb.py`):

```python
# 원본 임베딩 가중치 로드 [vocab_size, embed_dim]
emb = torch.load("embedding.pt")

# 컴파일 시 생성된 R1 회전 행렬 로드
rot = torch.jit.load("spinWeight/model/R1/global_rotation.pth")
rot_matrix = next(rot.parameters())

# 임베딩에 R1 회전 적용 (float64로 정밀도 유지 후 bfloat16으로 변환)
emb = (emb.double() @ rot_matrix.double()).bfloat16()
torch.save(emb, "embedding_rot.pt")
```

VLM 텍스트 임베딩 회전 예시 (`vlm/get_safetensors.py`):

```python
# HuggingFace safetensors에서 텍스트 임베딩 추출
with safe_open(SOURCE_FILE, framework="pt") as f:
    tensor = f.get_tensor("model.embed_tokens.weight")

# language 모델 컴파일 시 생성된 R1 회전 행렬 로드
rot_matrix = torch.jit.load(
    "spinWeight/Qwen2-VL-2B-Instruct_text_model/R1/global_rotation.pth"
).state_dict()["0"]

# 텍스트 임베딩에 R1 회전 적용
embedding = tensor.double() @ rot_matrix
save_file({"model.embed_tokens.weight": embedding.float()}, "mxq/model.safetensors")
```

**R2 (레이어별 회전)** 는 각 트랜스포머 레이어에 개별 회전을 적용하여
레이어 간 가중치 분포 차이를 보정합니다.
R2는 MXQ 컴파일 과정에서 모델 내부에 흡수되므로 별도 후처리가 필요 없습니다.

**VLM에서의 R1 활용**:
VLM의 경우 language 모델 컴파일 시 생성된 R1이 두 곳에서 사용됩니다.

1. **텍스트 임베딩 회전** — LLM과 동일하게 임베딩 가중치에 R1을 적용 (`vlm/get_safetensors.py`)
2. **비전 인코더 정렬** — vision encoder의 출력이 회전된 language 모델의 입력 공간과 일치해야 하므로,
   `HeadOutChRotation`으로 컴파일 시점에 R1을 참조 (`vlm/mxq_compile_vision.py`)

비전 임베딩 자체에는 별도 회전을 적용하지 않습니다.

**실제 사용 예시**:
- `llm/generate_mxq_4bit.py` - LLM 4bit SpinQuant 적용
- `llm/get_rotation_emb.py` - LLM 임베딩 R1 회전
- `vlm/mxq_compile_language.py` - VLM language 모델의 등가 변환
- `vlm/mxq_compile_vision.py` - VLM vision encoder에서 R1 회전 행렬 참조
- `vlm/get_safetensors.py` - VLM 텍스트 임베딩 R1 회전

---

## SearchWeightScaleConfig

레이어별 가중치 스케일을 학습하여 양자화 정확도를 보정합니다.
컴파일 시간이 길어지지만 양자화 모델의 정확도가 향상됩니다.
양자화 오차가 큰 4bit 양자화에서 사용을 권장합니다.

```python
from qbcompiler import SearchWeightScaleConfig

sws_config = SearchWeightScaleConfig(
    # 가중치 스케일 탐색 활성화.
    # calibration 데이터를 기반으로 각 레이어의 최적 가중치 스케일을
    # 반복 탐색하여, 양자화로 인한 정확도 저하를 최소화.
    apply=True,

    transformer=SearchWeightScaleConfig.Transformer(
        # 각 트랜스포머 컴포넌트별로 스케일 탐색 여부를 개별 지정.
        # True로 설정된 컴포넌트는 최적 스케일을 학습하여 양자화 오차를 줄임.
        # 컴포넌트를 많이 켤수록 정확도는 좋아지나 컴파일 시간이 비례하여 증가.
        query=True,   # Q projection 가중치 스케일 탐색
        key=True,     # K projection 가중치 스케일 탐색
        value=True,   # V projection 가중치 스케일 탐색
        out=True,     # Output projection 가중치 스케일 탐색
        ffn=True,     # FFN 가중치 스케일 탐색
    ),
)
```

**실제 사용 예시**:
- `llm/generate_mxq_4bit.py`

---

## PreprocessingConfig

calibration 데이터에 대한 이미지 전처리(resize, crop, normalize)를 컴파일러가 자동으로 수행합니다.

이 config를 적용하면 calibration 데이터를 별도로 전처리하지 않아도
raw 이미지를 그대로 `calib_data_path`에 전달할 수 있어 양자화 과정이 편리해집니다.

```python
from qbcompiler import PreprocessingConfig

preprocessing_config = PreprocessingConfig(
    # 전처리 파이프라인 활성화 여부
    apply=True,

    # True: 입력 이미지의 채널 포맷(RGB/BGR 등)을 자동으로 감지하고 변환.
    # 다양한 소스에서 온 이미지를 별도 변환 없이 처리할 수 있게 함.
    auto_convert_format=True,

    # 전처리 연산 파이프라인. 순서대로 적용됨.
    # 이 파이프라인이 모델의 첫 번째 레이어에 융합되어
    # NPU에서 전처리와 추론이 하나의 흐름으로 실행됨.
    pipeline=[
        # 1단계: 이미지를 256x256으로 리사이즈
        #   mode: 보간법 ("bilinear", "nearest" 등)
        {"op": "resize", "height": 256, "width": 256, "mode": "bilinear"},

        # 2단계: 중앙에서 224x224 크롭
        {"op": "centerCrop", "height": 224, "width": 224},

        # 3단계: 정규화
        {
            "op": "normalize",
            "mean": [0.485, 0.456, 0.406],  # ImageNet RGB 채널별 평균
            "std": [0.229, 0.224, 0.225],    # ImageNet RGB 채널별 표준편차

            # True: uint8 입력([0, 255])을 [0, 1]로 스케일링한 뒤 정규화.
            # 추론 시 원본 이미지를 바로 넣을 수 있게 함.
            "scaleToUint8": True,

            # True: 정규화 연산을 모델의 첫 번째 레이어 가중치에 흡수.
            # 별도 전처리 단계 없이 NPU 내에서 정규화가 처리됨.
            "fuseIntoFirstLayer": True,
        },
    ],
)
```

**실제 사용 예시**:
- `image_classification/model_compile.py`

---

## 다음 문서

- [Calibration 데이터 가이드](./02_about_calibration_data.KR.md) - 양자화에 사용되는 calibration 데이터 준비 방법
- [멀티 컴포넌트 모델 가이드](./03_about_multi_component.KR.md) - VLM/STT 등 분리 컴파일이 필요한 모델
