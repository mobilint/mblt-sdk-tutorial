# 대규모 언어 모델 (LLM) 런타임 추론

이 튜토리얼은 Mobilint NPU 하드웨어에서 컴파일된 Llama-3.2-1B-Instruct 모델로 추론을 실행하는 방법을 안내합니다.

이 튜토리얼에서는 [컴파일 튜토리얼](../../compilation/llm/README.md)에서 컴파일한 [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) 모델을 사용합니다. **먼저 컴파일 튜토리얼을 완료해야** 다음 파일들이 준비됩니다:

- `compilation/llm/Llama-3.2-1B-Instruct.mxq` - 컴파일된 모델
- `compilation/llm/embedding.pt` - 임베딩 레이어 가중치

## 개요

이 튜토리얼은 두 가지 추론 방법을 제공합니다:

| | 방법 A: mblt-model-zoo (권장) | 방법 B: 로컬 wrapper |
|---|---|---|
| 스크립트 | `inference_mblt_model_zoo.py` | `inference_mxq.py` |
| Step 1 필요 | 예 (`prepare_model.py`) | 아니오 (컴파일 출력에서 직접 로드) |
| NPU 관리 | 자동 (mblt-model-zoo 내부 처리) | 수동 (`qbruntime` 직접 호출) |
| API | HuggingFace `AutoModelForCausalLM` | 커스텀 `LlamaMXQ` 클래스 |
| NPU 코어 모드 설정 | `config.json` | wrapper 내 하드코딩 |

**방법 A**는 [mblt-model-zoo](https://docs.mobilint.com/model-zoo)를 사용하여 간편한 추론 방식을 제공합니다. 컴파일된 MXQ 모델을 한 줄(`AutoModelForCausalLM.from_pretrained()`)로 로드할 수 있으며, 일반 HuggingFace 모델과 동일한 방식으로 사용할 수 있습니다. NPU 코어 할당, KV cache, 리소스 관리는 자동으로 처리됩니다.

**방법 B**는 로컬 wrapper 클래스를 사용하여 `qbruntime`을 직접 호출하며, 모델 로딩에 mblt-model-zoo에 의존하지 않습니다.

모든 스크립트는 `runtime/llm/` 디렉토리에서 실행합니다.

## 사전 요구 사항

```bash
pip install -r requirements.txt
```

## Step 1: 모델 폴더 준비

컴파일된 MXQ 파일을 복사하고 HuggingFace에서 필요한 config와 토크나이저를 다운로드합니다.

```bash
python prepare_model.py \
    --mxq-path ../../compilation/llm/Llama-3.2-1B-Instruct.mxq \
    --embedding-path ../../compilation/llm/embedding.pt \
    --output-folder ./llama-mxq \
    --model-id mobilint/Llama-3.2-1B-Instruct
```

**수행 내용:**

- 컴파일된 MXQ 파일을 출력 폴더로 복사
- 임베딩 가중치를 `model.safetensors`로 복사
- HuggingFace에서 config.json과 토크나이저 다운로드
- config.json에 NPU 코어 할당 설정(`target_cores`) 추가

**출력:**

- `./llama-mxq/` - 추론 준비가 완료된 모델 폴더

## Step 2: 추론 실행

두 가지 추론 방법을 사용할 수 있습니다:

### 방법 A: mblt-model-zoo 사용 (권장)

mblt-model-zoo의 공식 Llama 구현을 사용하여 HuggingFace `AutoModel` API로 MXQ 모델을 로드합니다. `prepare_model.py`(Step 1) 실행이 필요합니다.

```bash
python inference_mblt_model_zoo.py \
    --model-folder ./llama-mxq \
    --model-id mobilint/Llama-3.2-1B-Instruct
```

**추가 옵션:**

```bash
# 커스텀 프롬프트
python inference_mblt_model_zoo.py --model-folder ./llama-mxq --model-id mobilint/Llama-3.2-1B-Instruct --prompt "양자 컴퓨팅이란 무엇인가요?"

# 더 긴 생성
python inference_mblt_model_zoo.py --model-folder ./llama-mxq --model-id mobilint/Llama-3.2-1B-Instruct --max-new-tokens 512
```

### 방법 B: 로컬 wrapper 사용

로컬 wrapper 클래스(`wrapper/llama_model.py`)를 사용하여 `qbruntime`을 직접 호출합니다. `prepare_model.py`가 필요 없으며 컴파일 출력에서 직접 로드합니다.

```bash
python inference_mxq.py \
    --mxq-path ../../compilation/llm/Llama-3.2-1B-Instruct.mxq \
    --embedding-weight-path ../../compilation/llm/embedding.pt
```

**추가 옵션:**

```bash
# 커스텀 프롬프트
python inference_mxq.py --prompt "양자 컴퓨팅이란 무엇인가요?"

# 더 긴 생성
python inference_mxq.py --max-new-tokens 512
```

## NPU 추론 모드

NPU는 다양한 코어 모드를 지원합니다. 코어 모드는 `config.json`(`prepare_model.py`가 생성)에서 설정합니다.

| 모드 | 설명 | config.json 필드 |
|------|------|-----------------|
| `single` | 각 코어가 독립적으로 추론 실행. 기본 모드. | `target_cores: ["0:0"]` |
| `multi` | 여러 코어가 하나의 추론에 협력. | `core_mode: "multi"`, `target_clusters: [0]` |
| `global4` | 4개 코어(1 클러스터)가 글로벌 모드로 실행. | `core_mode: "global4"`, `target_clusters: [0]` |
| `global8` | 8개 코어(2 클러스터)가 글로벌 모드로 실행. 최대 처리량. | `core_mode: "global8"`, `target_clusters: [0, 1]` |

## 커맨드 라인 인자

### `prepare_model.py`

| 인자 | 기본값 | 설명 |
| --- | ----- | --- |
| `--mxq-path` | `../../compilation/llm/Llama-3.2-1B-Instruct.mxq` | 컴파일된 MXQ 파일 경로 |
| `--embedding-path` | `../../compilation/llm/embedding.pt` | 임베딩 가중치 파일 경로 |
| `--output-folder` | `./llama-mxq` | 준비된 모델의 저장 폴더 |
| `--model-id` | `mobilint/Llama-3.2-1B-Instruct` | config 및 토크나이저 다운로드용 HuggingFace 모델 ID |

### `inference_mblt_model_zoo.py`

| 인자 | 기본값 | 설명 |
| --- | ----- | --- |
| `--model-folder` | `./llama-mxq` | 준비된 모델 폴더 경로 |
| `--model-id` | `mobilint/Llama-3.2-1B-Instruct` | 토크나이저 다운로드용 HuggingFace 모델 ID |
| `--prompt` | `"Explain the concept of NPU..."` | 모델에 전달할 프롬프트 |
| `--max-new-tokens` | `256` | 최대 생성 토큰 수 |

### `inference_mxq.py`

| 인자 | 기본값 | 설명 |
| --- | ----- | --- |
| `--mxq-path` | `../../compilation/llm/Llama-3.2-1B-Instruct.mxq` | 컴파일된 MXQ 파일 경로 |
| `--embedding-weight-path` | `../../compilation/llm/embedding.pt` | 임베딩 가중치 파일 경로 |
| `--prompt` | `"Explain the concept of NPU..."` | 모델에 전달할 프롬프트 |
| `--max-new-tokens` | `256` | 최대 생성 토큰 수 |

## 파일 구조

```text
llm/
├── prepare_model.py              # Step 1: 모델 폴더 준비
├── inference_mblt_model_zoo.py   # Step 2: 방법 A (mblt-model-zoo)
├── inference_mxq.py              # Step 2: 방법 B (로컬 wrapper)
├── wrapper/
│   └── llama_model.py            # qbruntime 직접 호출 로컬 wrapper
└── llama-mxq/                    # Step 1의 출력
    ├── config.json               # NPU 코어 할당이 포함된 모델 구성
    ├── model.safetensors         # 임베딩 가중치
    ├── Llama-3.2-1B-Instruct.mxq
    └── ...
```

## 참고 사항

- 임베딩 레이어는 CPU에서 실행되고 트랜스포머 레이어는 NPU에서 실행됩니다
- 텍스트 생성은 HuggingFace의 표준 `generate()` API를 스트리밍 출력과 함께 사용합니다

## 참조

- [컴파일 튜토리얼](../../compilation/llm/README.md)
- [Llama-3.2-1B-Instruct 모델 카드](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [mblt-model-zoo Documentation](https://docs.mobilint.com/model-zoo)
- [Mobilint Documentation](https://docs.mobilint.com)
