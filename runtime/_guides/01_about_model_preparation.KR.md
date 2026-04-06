# 모델 준비 가이드

`mblt-model-zoo` API를 사용하려면 컴파일 결과물을 모델 폴더로 구성해야 합니다.
이 문서는 `prepare_model.py`가 수행하는 작업과 `config.json`의 구조를 설명합니다.

> 단순 모델(Image Classification, Object Detection 등)은
> `.mxq` 파일을 `qbruntime`에 직접 넘기므로 이 준비 과정이 필요 없습니다.

---

## prepare_model.py가 하는 일

각 튜토리얼 디렉토리에 제공되는 `prepare_model.py`는
컴파일 결과물을 `mblt-model-zoo`가 인식하는 폴더 구조로 변환합니다.

수행 작업:

1. **`.mxq` 파일 복사** — 컴파일 결과물을 출력 폴더로 복사
2. **임베딩 가중치 변환** — `.pt` → safetensors 형식으로 변환 (모델에 따라)
3. **config.json 구성** — `mxq_path`, `_name_or_path`, `target_cores` 등 NPU 설정 추가
4. **tokenizer 다운로드** — HuggingFace에서 tokenizer 파일 다운로드 (텍스트 모델의 경우)

**실제 사용 예시**:
- `llm/prepare_model.py`
- `vlm/prepare_model.py`
- `stt/prepare_model.py`

---

## config.json 구조

`config.json`은 HuggingFace 표준 config에 NPU 전용 필드가 추가된 형태입니다.

### 공통 필드

```json
{
    // HuggingFace 표준 필드
    "architectures": ["MobilintLlamaForCausalLM"],
    "model_type": "mobilint-llama",

    // mblt-model-zoo가 모델을 자동 인식하기 위한 ID
    "_name_or_path": "mobilint/Llama-3.2-1B-Instruct",

    // 컴파일된 MXQ 파일 경로 (모델 폴더 기준 상대 경로)
    "mxq_path": "Llama-3.2-1B-Instruct.mxq",

    // NPU 코어 할당
    "target_cores": ["0:0"]
}
```

### 멀티 컴포넌트 모델

서브모델이 여러 개인 경우, 각 서브모델의 `mxq_path`와 `target_cores`가 별도로 지정됩니다.

**VLM** (vision + language):

```json
{
    "mxq_path": "text_model.mxq",
    "target_cores": ["0:0"],
    "vision_config": {
        "mxq_path": "vision_transformer.mxq",
        "target_cores": ["0:0"]
    }
}
```

**STT** (encoder + decoder):

```json
{
    "encoder_mxq_path": "whisper_encoder.mxq",
    "encoder_target_cores": ["0:0"],
    "decoder_mxq_path": "whisper_decoder.mxq",
    "decoder_target_cores": ["0:0"]
}
```

---

## NPU 코어 할당

`target_cores`는 모델을 어떤 NPU 코어에서 실행할지 지정합니다.

> 코어 모드에 관한 상세한 설명은
> [Mobilint Multi-Core Documentation](https://docs.mobilint.com/v1.0/en/multicore.html)을 참조하세요.

### 코어 모드

| 모드 | 설정 | 설명 |
|------|------|------|
| single | `"target_cores": ["0:0"]` | 단일 코어에서 실행 |
| multi | `"core_mode": "multi", "target_clusters": [0]` | 한 클러스터 내 여러 코어 협업 |
| global4 | `"core_mode": "global4", "target_clusters": [0]` | 한 클러스터의 4개 코어 사용 |
| global8 | `"core_mode": "global8", "target_clusters": [0, 1]` | 두 클러스터의 8개 코어 사용 |

### target_cores 포맷

`"cluster:core"` 형태로 지정합니다.

```text
"0:0" → Cluster 0, Core 0
"0:1" → Cluster 0, Core 1
"1:0" → Cluster 1, Core 0
```

### multi/global 모드 사용 시

`target_cores` 대신 `core_mode`와 `target_clusters`를 사용합니다.

```json
{
    "core_mode": "global4",
    "target_clusters": [0]
}
```

> 컴파일 시 `inference_scheme="all"`로 컴파일했다면 모든 코어 모드를 사용할 수 있습니다.
> 특정 scheme으로 컴파일한 경우 해당 모드만 사용 가능합니다.

---

## 다음 문서

- [런타임 파이프라인 개요](./00_about_runtime_pipeline.KR.md) - 전체 흐름
- [추론 API 가이드](./02_about_inference_api.KR.md) - qbruntime vs mblt-model-zoo
