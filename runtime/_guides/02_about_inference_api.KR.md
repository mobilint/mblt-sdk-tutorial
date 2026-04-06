# 추론 API 가이드

Mobilint NPU에서 모델을 추론하는 두 가지 API를 소개합니다.

> 컴파일된 `.mxq` 파일이 준비되어 있다는 전제 하에 설명합니다.
> 컴파일 과정은 [컴파일 파이프라인 개요](../../compilation/_guides/00_about_compilation_pipeline.KR.md)를 참조하세요.

---

## API 종류

| API | 수준 | 특징 | 적합한 모델 |
|-----|------|------|------------|
| `qbruntime` | low-level | NPU 직접 제어, numpy 입출력 | 단순 모델 (Image Classification, Object Detection, BERT 등) |
| `mblt-model-zoo` | high-level | HuggingFace 호환 API, 자동 NPU 관리 | 복합 모델 (LLM, VLM, STT 등) |

---

## qbruntime

NPU를 직접 제어하는 low-level API입니다.
`.mxq` 파일을 로드하고, numpy 배열을 입력으로 넘겨 추론합니다.

```python
import numpy as np
import qbruntime

# 1. NPU accelerator 생성
#    인자는 device 번호 (0번부터 시작)
acc = qbruntime.Accelerator(0)

# 2. 모델 설정
mc = qbruntime.ModelConfig()

# 코어 할당 설정.
# CoreId(cluster, core)로 특정 코어를 지정.
mc.set_single_core_mode(
    None,
    [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)]
)

# 3. 모델 로드 및 NPU에 배치
model = qbruntime.Model("./model.mxq", mc)
model.launch(acc)

# 4. 추론 실행
#    입력: numpy 배열 (dtype, shape은 모델에 따라 다름)
#    출력: numpy 배열의 리스트
output = model.infer(input_numpy_array)

# 5. 리소스 정리
model.dispose()
```

### 입력 데이터 포맷

컴파일 시 `PreprocessingConfig`로 전처리를 융합한 모델은
**UInt8 형식의 raw 이미지**를 입력으로 받습니다.

```python
# 이미지를 numpy uint8 배열로 변환
image = np.array(pil_image, dtype=np.uint8)  # [H, W, C]
output = model.infer(image)
```

별도의 정규화(normalize)가 불필요합니다.

**실제 사용 예시**:
- `image_classification/inference_mxq.py`
- `object_detection/inference_mxq.py`
- `pose_estimation/inference_mxq.py`

---

## mblt-model-zoo

HuggingFace `transformers` API와 호환되는 high-level API입니다.
모델 폴더의 `config.json`에 포함된 `mxq_path`, `target_cores`, `_name_or_path` 등의 설정을 기반으로
모델을 자동 로드하고 NPU 리소스를 관리합니다.

> config.json의 구조에 대해서는
> [모델 준비 가이드](./01_about_model_preparation.KR.md)를 참조하세요.

```python
# mblt-model-zoo의 모델 구현을 import하여 HuggingFace에 등록
# 이 import가 없으면 config.json의 architectures를 인식하지 못함
import mblt_model_zoo.hf_transformers.models.llama.modeling_llama  # noqa: F401

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# 1. 모델 로드
#    config.json에서 mxq_path, target_cores 등을 자동 읽음
model = AutoModelForCausalLM.from_pretrained("./model-folder")

# 2. 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("model-id")

# 3. 추론 실행
#    HuggingFace 표준 generate() API 사용
inputs = tokenizer(["prompt text"], return_tensors="pt")
streamer = TextStreamer(tokenizer, skip_prompt=True)

output = model.generate(
    **inputs,
    max_new_tokens=256,
    streamer=streamer,      # 토큰이 생성될 때마다 실시간 출력
)

# 4. 리소스 정리
model.dispose()
```

### mblt-model-zoo import 규칙

`from_pretrained()`이 config.json의 `architectures` 필드를 인식하려면,
해당 모델 구현을 사전에 import해야 합니다.

```python
# LLM
import mblt_model_zoo.hf_transformers.models.llama.modeling_llama

# VLM
import mblt_model_zoo.hf_transformers.models.qwen2_vl.modeling_qwen2_vl

# STT
import mblt_model_zoo.hf_transformers.models.whisper.modeling_whisper
```

**실제 사용 예시**:
- `llm/inference_mblt_model_zoo.py`
- `vlm/inference_mblt_model_zoo.py`
- `stt/inference_mblt_model_zoo.py`

---

## Split Execution (CPU/NPU 분리)

복합 모델에서는 모든 레이어가 NPU에서 실행되지 않습니다.
임베딩 레이어는 토큰 ID로 가중치 테이블에서 벡터를 꺼내오는 룩업 연산이므로 CPU에서 실행됩니다.

어떤 레이어가 CPU/NPU에서 실행되는지는 모델과 컴파일 구성에 따라 다릅니다.

이 분리는 `mblt-model-zoo`에서 자동 처리되며,
`qbruntime`을 직접 사용하는 경우 wrapper 클래스에서 구현합니다.

> wrapper 클래스 예시: `bert/wrapper/bert_model.py`, `llm/wrapper/llama_model.py`

---

## KV Cache 관리

transformer decoder 구조의 모델(LLM, STT decoder)은
autoregressive 생성 시 이전 토큰의 key/value를 캐시합니다.

NPU 내부에서 KV cache가 관리되며, 새로운 생성을 시작할 때 캐시를 초기화해야 합니다.

```python
# mblt-model-zoo 사용 시 dispose()가 캐시도 함께 정리
model.dispose()

# qbruntime wrapper 사용 시 명시적 캐시 초기화
model.mxq_model.dump_cache_memory()
```

---

## 다음 문서

- [런타임 파이프라인 개요](./00_about_runtime_pipeline.KR.md) - 전체 흐름
- [모델 준비 가이드](./01_about_model_preparation.KR.md) - config.json 구조와 NPU 코어 할당
