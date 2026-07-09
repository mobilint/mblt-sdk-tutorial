# LLM 런타임

이 튜토리얼은 Mobilint NPU 하드웨어에서 컴파일된 `Llama-3.2-1B-Instruct` 모델을 실행하는 방법을 설명합니다.

시작하기 전에 [../../../compilation/llm/README.KR.md](../../../compilation/llm/README.KR.md)의 컴파일 과정을 먼저 완료하세요. 이 디렉토리의 런타임 예제는 다음 파일이 준비되어 있다고 가정합니다.

- `../../../compilation/llm/Llama-3.2-1B-Instruct.mxq`
- `../../../compilation/llm/embedding.pt`

## 사전 준비

필요한 패키지를 설치하세요.

```bash
pip install -r requirements.txt
```

## 개요

이 튜토리얼은 두 가지 런타임 경로를 제공합니다.

- `inference_mblt_model_zoo.py`: 권장 경로. `mblt-model-zoo`와 Hugging Face 스타일 API를 사용합니다.
- `inference_mxq.py`: 로컬 래퍼 경로. `wrapper/llama_model.py`를 통해 `qbruntime`을 직접 호출합니다.

권장 경로는 준비된 모델 폴더가 필요하며, 로컬 래퍼 경로는 컴파일 결과를 직접 읽을 수 있습니다.

## 이 튜토리얼의 파일

- `prepare_model.py`: `mblt-model-zoo`용 모델 폴더를 생성합니다.
- `inference_mblt_model_zoo.py`: `mblt-model-zoo`를 통해 텍스트 생성을 실행합니다.
- `inference_mxq.py`: 로컬 `LlamaMXQ` 래퍼를 통해 텍스트 생성을 실행합니다.
- `wrapper/llama_model.py`: CPU 측 임베딩과 NPU 실행을 결합한 로컬 래퍼입니다.
- `requirements.txt`: 이 튜토리얼에 필요한 Python 의존성입니다.

## Step 1: 모델 폴더 준비

권장되는 `mblt-model-zoo` 흐름을 사용하려면 먼저 이 단계를 실행하세요.

```bash
python prepare_model.py \
    --mxq-path ../../../compilation/llm/Llama-3.2-1B-Instruct.mxq \
    --embedding-path ../../../compilation/llm/embedding.pt \
    --output-folder ./llama-mxq \
    --model-id mobilint/Llama-3.2-1B-Instruct
```

이 스크립트는 다음 작업을 수행합니다.

- 컴파일된 MXQ 파일을 출력 폴더로 복사
- `embedding.pt`를 `model.safetensors`로 변환
- Hugging Face에서 토크나이저와 설정 파일 다운로드
- `config.json`에 기본 NPU 코어 할당 설정 추가

## Step 2A: `mblt-model-zoo`로 추론 실행

이 경로를 권장합니다.

```bash
python inference_mblt_model_zoo.py \
    --model-folder ./llama-mxq \
    --model-id mobilint/Llama-3.2-1B-Instruct
```

자주 쓰는 옵션:

```bash
python inference_mblt_model_zoo.py --model-folder ./llama-mxq --model-id mobilint/Llama-3.2-1B-Instruct --prompt "양자 컴퓨팅이란 무엇인가요?"
python inference_mblt_model_zoo.py --model-folder ./llama-mxq --model-id mobilint/Llama-3.2-1B-Instruct --max-new-tokens 512
```

## Step 2B: 로컬 래퍼로 추론 실행

이 경로는 컴파일 결과를 직접 읽으므로 `prepare_model.py`가 필요하지 않습니다.

```bash
python inference_mxq.py \
    --mxq-path ../../../compilation/llm/Llama-3.2-1B-Instruct.mxq \
    --embedding-weight-path ../../../compilation/llm/embedding.pt
```

자주 쓰는 옵션:

```bash
python inference_mxq.py --prompt "양자 컴퓨팅이란 무엇인가요?"
python inference_mxq.py --max-new-tokens 512
```

## NPU 코어 모드

생성된 `config.json`을 편집해 NPU 코어 사용 방식을 바꿀 수 있습니다.

| 모드 | 설명 | 예시 설정 필드 |
| --- | --- | --- |
| `single` | 단일 코어 실행 | `target_cores: ["0:0"]` |
| `multi` | 여러 코어가 하나의 추론에 협력 | `core_mode: "multi"`, `target_clusters: [0]` |
| `global4` | 한 클러스터를 글로벌 모드로 사용 | `core_mode: "global4"`, `target_clusters: [0]` |
| `global8` | 두 클러스터를 글로벌 모드로 사용 | `core_mode: "global8"`, `target_clusters: [0, 1]` |

## 파라미터

### `prepare_model.py`

- `--mxq-path`: 컴파일된 MXQ 파일 경로
- `--embedding-path`: 임베딩 가중치 파일 경로
- `--output-folder`: 준비된 모델 저장 폴더
- `--model-id`: 설정 파일과 토크나이저 다운로드에 사용할 Hugging Face 모델 ID

### `inference_mblt_model_zoo.py`

- `--model-folder`: 준비된 모델 폴더 경로
- `--model-id`: 토크나이저 다운로드에 사용할 Hugging Face 모델 ID
- `--prompt`: 사용자 프롬프트
- `--max-new-tokens`: 최대 생성 토큰 수

### `inference_mxq.py`

- `--mxq-path`: 컴파일된 MXQ 파일 경로
- `--embedding-weight-path`: 임베딩 가중치 파일 경로
- `--prompt`: 사용자 프롬프트
- `--max-new-tokens`: 최대 생성 토큰 수

## 예상 출력

두 추론 경로 모두 입력한 프롬프트에 대한 생성 텍스트를 스트리밍 형태로 출력합니다.

이 튜토리얼에서는 토큰 임베딩은 CPU에서, 트랜스포머 레이어는 Mobilint NPU에서 실행됩니다.
