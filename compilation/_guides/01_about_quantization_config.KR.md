# 컴파일 Config 가이드

`mxq_compile()`에 전달하는 config 객체들을 소개합니다.

양자화 config는 파라미터 조합이 다양하며, 모델 아키텍처와 태스크에 따라 최적의 값이 다릅니다.
동일한 설정이라도 모델에 따라 정확도와 성능에 미치는 영향이 달라질 수 있으므로,
각 튜토리얼 디렉토리(`image_classification/`, `llm/`, `vlm/` 등)에서 유사한 모델이나 태스크의
컴파일 스크립트를 참고하여 베이스라인 config로 활용하는 것을 권장합니다.

---

## Config 종류 overview

| Config | 역할 | 사용 대상 |
| -------- | ------ | ---------- |
| `CalibrationConfig` | quantization range 결정 방식 (per-channel, percentile 등) | quantization이 필요한 모든 모델 |
| `BitConfig` | transformer component별 quantization bit 수 (8bit/4bit) | transformer 기반 모델 (LLM 등) |
| `LlmConfig` | sequence length, KV cache, NPU core 할당 | transformer decoder 구조 (autoregressive + KV cache) |
| `EquivalentTransformationConfig` | SpinQuant 등 quantization 오차를 줄이는 고급 수학적 변환 | 4bit quantization에서 사용 권장 |
| `SearchWeightScaleConfig` | layer별 weight scale 학습으로 quantization 정확도 보정 | 4bit quantization에서 사용 권장 |
| `HessianQuantConfig` | Hessian 기반 weight rounding(GPTQ 계열)과 solver 선택 | 선택 사항, LLM / VLM decoder |
| `BiasCorrectionConfig` | quantized convolution의 채널별 계통 오차 보정 | 모든 모델 (1.4부터 기본 활성화) |
| `ResourceManagementConfig` | weight dtype, weight 메모리 방식, quantization 중 GPU 메모리 예산 | 대형 모델 |
| `PreprocessingConfig` | calibration 진행 시 image 전처리를 compiler가 자동 수행 | image 입력 모델 (Image Classification 등) |

> **qbcompiler 1.4에서는 기본 결과가 달라집니다.**
> bias correction이 기본으로 켜지고, weight quantization이 각 layer를 이전 layer들의 quantized 출력 기준으로 풉니다.
> 따라서 같은 모델과 config라도 qbcompiler 1.3과 다른 MXQ가 만들어지며, 정확도가 다르고 컴파일 시간이 길어질 수 있습니다.
> 1.3으로 측정한 정확도 수치는 다시 측정해야 합니다.

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
- `llm/compile_model.py`
- `bert/compile_model.py`

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

- `llm/compile_model.py` - 8bit
- `llm/compile_model_4bit.py` - W4V8

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

- `llm/compile_model.py` - LLM 컴파일 시 시퀀스/캐시 길이 설정
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

- `llm/compile_model_4bit.py` - LLM 4bit SpinQuant 적용
- `vlm/compile_decoder.py` - VLM decoder의 등가 변환
- `vlm/compile_encoder.py` - VLM encoder에서 R1 회전 행렬 참조 (`HeadOutChRotation`)

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
이 회전은 컴파일된 MXQ 모델 내부에 이미 반영됩니다. **임베딩 레이어는 MXQ에 포함되지 않고 CPU에서 실행**되므로 추론에는 동일하게 R1 회전된 임베딩 가중치가 필요합니다.

- SpinQuant(R1)를 사용하지 않는 경우: 임베딩 회전 불필요
- SpinQuant(R1)를 사용하는 경우: 임베딩에 R1 회전 필수

LLM 튜토리얼은 Mobilint Hugging Face 저장소에 있는 회전 완료된 `model.safetensors`를 재사용합니다.

VLM 텍스트 임베딩 회전 예시 (`vlm/prepare_model.py`):

```python
# HuggingFace safetensors에서 텍스트 임베딩 추출
with safe_open(SOURCE_FILE, framework="pt") as f:
    tensor = f.get_tensor("model.language_model.embed_tokens.weight")

# language 모델 컴파일 시 생성된 R1 회전 행렬 로드
rot_matrix = torch.jit.load(
    "spinWeight/aries-rb/global_rotation.pth"
).state_dict()["0"]

# 텍스트 임베딩에 R1 회전 적용
embedding = tensor.double() @ rot_matrix
save_file({"model.language_model.embed_tokens.weight": embedding.float()}, "prepared/model.safetensors")
```

**R2 (레이어별 회전)** 는 각 트랜스포머 레이어에 개별 회전을 적용하여
레이어 간 가중치 분포 차이를 보정합니다.
R2는 MXQ 컴파일 과정에서 모델 내부에 흡수되므로 별도 후처리가 필요 없습니다.

**VLM에서의 R1 활용**:
VLM의 경우 language 모델 컴파일 시 생성된 R1이 두 곳에서 사용됩니다.

1. **텍스트 임베딩 회전** — LLM과 동일하게 임베딩 가중치에 R1을 적용 (`vlm/prepare_model.py`)
2. **비전 인코더 정렬** — vision encoder의 출력이 회전된 language 모델의 입력 공간과 일치해야 하므로,
   `HeadOutChRotation`으로 컴파일 시점에 R1을 참조 (`vlm/compile_encoder.py`)

비전 임베딩 자체에는 별도 회전을 적용하지 않습니다.

**실제 사용 예시**:

- `llm/compile_model_4bit.py` - LLM 4bit SpinQuant 적용
- `llm/prepare_model.py` - LLM 회전 임베딩 재사용
- `vlm/compile_decoder.py` - VLM decoder의 등가 변환
- `vlm/compile_encoder.py` - VLM encoder에서 R1 회전 행렬 참조
- `vlm/prepare_model.py` - VLM 텍스트 임베딩 R1 회전 및 런타임 패키징

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

- `llm/compile_model_4bit.py`

---

## HessianQuantConfig

calibration activation으로 계산한 Hessian 기반 solver(GPTQ 계열)로 weight를 rounding합니다. weight를 하나씩 독립적으로 rounding하는 대신 오차를 함께 보정합니다.
컴파일 시간과 메모리가 더 들며, 일부 decoder 컴포넌트에서 사용합니다(예: REGULUS용 `vlm/compile_config.py`, `stt/compile_config.py`).

```python
from qbcompiler import HessianQuantConfig

hessian_quant_config = HessianQuantConfig(
    apply=True,

    # layer별 정수 weight를 계산하는 solver (1.4 신규).
    #   "symmetric"         - 기본값. quantized 입력 기준 오차를 최소화.
    #   "asymmetric_causal" - 원본 float 출력을 목표로 함. 각 column의 입력 불일치를
    #                         뒤쪽 column에만 alpha 배율로 전달.
    #   "asymmetric_refit"  - 원본 float 출력을 목표로 함. 최소제곱으로 weight를 다시 맞춘 뒤
    #                         그 주변에서 symmetric solve 수행.
    solver="symmetric",

    # Residual compensation (1.4 신규): block 사이에 전파되며 누적되는
    # weight drift를 추가로 보정.
    rescomp=False,

    attributes=HessianQuantConfig.Attributes(
        act_order=True,     # activation 크기 순으로 column 처리
        block_size=128,     # block당 solve할 column 수 (기본값 256)
        perc_damp=0.01,     # Hessian 대각에 더하는 damping
        alpha=0.25,         # asymmetric_causal / rescomp 보정 강도
    ),
)
```

Group-wise Hessian quantization은 `solver="symmetric"`, `rescomp=False` 조합만 허용하며 다른 조합은 거부됩니다.

**실제 사용 예시**:

- `vlm/compile_config.py` - REGULUS용 VLM decoder
- `stt/compile_config.py` - Whisper decoder

---

## BiasCorrectionConfig

weight quantization 중에 각 quantized convolution의 채널별 출력 오차를 calibration 샘플에서 원본 float weight와 비교해 측정하고,
그 보정값을 정수 bias에 반영합니다.
보정은 한 번의 패스로 layer 순서대로 전파됩니다.

qbcompiler 1.4부터 bias correction은 **기본으로 활성화**되므로 이 저장소의 튜토리얼은 별도로 설정하지 않습니다.
이전 버전의 `LayerBiasCorrectionConfig` / `layer_bias_correction`을 대체하며, 기존의 `numSamples`, `iterations`, `correctionRate` 속성은 없어졌습니다.

```python
from qbcompiler import BiasCorrectionConfig, mxq_compile

# 끄기 (예: qbcompiler 1.3 빌드와 비교할 때)
mxq_compile(..., bias_correction=False)

# 또는 일부 layer로 제한
mxq_compile(
    ...,
    bias_correction_config=BiasCorrectionConfig(
        apply=True,
        attributes=BiasCorrectionConfig.Attributes(apply_layers=[], exclude_layers=[]),
    ),
)
```

---

## ResourceManagementConfig

quantization 중 메모리 사용을 제어합니다.

```python
from qbcompiler import ResourceManagementConfig

resource_management_config = ResourceManagementConfig(
    # calibration 중 사용하는 weight dtype.
    weight_dtype="float32",

    # layer를 quantize하는 동안 float weight를 보관하는 방식 (0-4:
    # DeleteFloat, SaveFloat, MoveFloat, KeepFloat, KeepAll).
    weight_memory=ResourceManagementConfig.WeightMemory(method=1),

    # weight quantization의 GPU 메모리 예산, MiB 단위 (1.4 신규).
    #   -1: 자동 (기본값), 0: 제한 없음.
    # 복구 가능한 CUDA out-of-memory 오류가 나면 더 작은 batch로 재시도함.
    gpu_memory_budget_mb=-1,
)
```

`use_gpu_only_for_calibration`(JSON config의 `useGPUOnlyForCalibration`)은 qbcompiler 1.4에서 제거되었으며, 이 항목이 남아 있는 config는 거부됩니다.
기존 config에서 삭제하세요. GPU 메모리 사용은 이제 `gpu_memory_budget_mb`가 관리합니다.

**실제 사용 예시**:

- `vlm/compile_config.py`
- `mask_generation/compile_config.json`

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

    # 전처리 연산 파이프라인. calibration 이미지에 순서대로 적용됨.
    # "fuseIntoFirstLayer"가 지정된 연산만 컴파일된 모델에 포함되고,
    # 나머지(resize, centerCrop 등)는 calibration 이미지를 준비하는 방법을 기술하므로
    # 애플리케이션은 추론 전에 같은 단계를 직접 적용해야 함.
    pipeline=[
        # 1단계: 이미지를 256x256으로 리사이즈
        #   mode: 보간법 ("bilinear", "nearest" 등)
        #   backend (1.4 신규): "torch" (기본값), "pil", "opencv" — 평가나 애플리케이션에서
        #   쓰는 라이브러리를 선택하면 calibration이 같은 픽셀을 보게 됨.
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

qbcompiler 1.4 참고 사항:

- 각 연산은 자신의 key만 받으며, 알 수 없는 key는 무시되지 않고 오류가 됩니다.
  예를 들어 `padValue` 대신 `padvalue`처럼 잘못 쓴 key는 해당 연산이 받는 key 목록과 함께 컴파일 오류를 냅니다.
- `resize`와 `letterbox`는 Pillow나 OpenCV 전처리와 정확히 맞추기 위해 `backend: "pil"` 또는 `"opencv"`를 받습니다. `alignCorners`와 `antialias`는 기본 `torch` backend에만 적용되며 다른 backend와 함께 쓰면 거부됩니다.
- `letterbox.alignType`은 가운데 정렬 패딩(`0`, 기본값) 또는 왼쪽 위 배치(`1`)를 선택합니다.
- `letterbox.fuseIntoFirstLayer`는 지원되는 정수 배율 다운샘플링을 첫 convolution에 접어 넣어, 모델이 선언한 `sourceHeight` x `sourceWidth` 해상도를 직접 받게 할 수도 있습니다. `torch` 또는 `opencv` backend가 필요하며 calibration 이미지도 그 해상도여야 합니다.
- `classification_torchvision` preset은 이제 Pillow로 resize하고 `yolo_640` / `yolo_1280` preset은 OpenCV로 letterbox하므로, calibration 텐서가 이전 버전과 다를 수 있습니다.

**실제 사용 예시**:

- `image_classification/model_compile.py`

---

## 다음 문서

- [Calibration 데이터 가이드](./02_about_calibration_data.KR.md) - 양자화에 사용되는 calibration 데이터 준비 방법
- [멀티 컴포넌트 모델 가이드](./03_about_multi_component.KR.md) - VLM/STT 등 분리 컴파일이 필요한 모델
