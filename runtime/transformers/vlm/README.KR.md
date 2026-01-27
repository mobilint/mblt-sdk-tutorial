# 비전 언어 모델 (VLM) 런타임 추론 (Vision Language Model (VLM) Runtime Inference)

이 튜토리얼은 Mobilint 런타임을 사용하여 Aries 2 하드웨어에서 컴파일된 Qwen2-VL-2B-Instruct 모델로 추론을 실행하는 방법에 대한 지침을 제공합니다.

## 개요 (Overview)

[컴파일 튜토리얼](../../../compilation/transformers/vlm/README.md)에 따라 VLM 모델을 컴파일한 후, 제공된 런타임 스크립트를 사용하여 컴파일된 MXQ 모델로 추론을 실행할 수 있습니다.

추론 스크립트는 다음 방법을 보여줍니다:

- 컴파일 출력 디렉토리에서 컴파일된 MXQ 모델 로드.
- 이미지-텍스트-대-텍스트(image-text-to-text) 작업을 위한 Mobilint 런타임 파이프라인 초기화.
- 이미지와 텍스트가 모두 포함된 비전-언어 쿼리 처리.
- 실시간 스트리밍 출력을 통한 추론 실행.

## 사전 요구 사항 (Prerequisites)

추론을 실행하기 전에 다음 사항을 확인하십시오:

- **컴파일 튜토리얼 완료**: 필수 파일 4개가 모두 컴파일 디렉토리에 존재해야 합니다.
- **`mblt-model-zoo` 패키지 설치**:

  ```bash
  pip install mblt-model-zoo
  ```

- **필수 종속성 설치**:

  ```bash
  pip install transformers==4.54.0 torch pillow
  ```

## 필수 파일 (Required Files)

추론 스크립트는 컴파일 출력 디렉토리(런타임 디렉토리 기준 `../../../compilation/transformers/vlm/compile/mxq/`)에 다음 4개의 파일이 존재할 것으로 예상합니다:

1.  **Qwen2-VL-2B-Instruct_text_model.mxq**: 컴파일된 언어 모델.
2.  **Qwen2-VL-2B-Instruct_vision_transformer.mxq**: 컴파일된 비전 인코더.
3.  **config.json**: MXQ 경로를 포함한 모델 구성 파일.
4.  **model.safetensors**: 회전된(rotated) 임베딩 가중치.

이 파일들은 컴파일 튜토리얼을 성공적으로 완료하면 자동으로 생성됩니다.

## 추론 실행 (Running Inference)

### 기본 사용법 (Basic Usage)

추론 스크립트를 실행하려면:

```bash
cd /workspace/mblt-sdk-tutorial/runtime/transformers/vlm
python run_qwen2_vl_local.py
```

스크립트는 다음 작업을 수행합니다:

1.  지정된 디렉토리에서 컴파일된 MXQ 모델을 로드합니다.
2.  Hugging Face에서 프로세서를 로드합니다 (모델 ID: `mobilint/Qwen2-VL-2B-Instruct`).
3.  샘플 프롬프트를 사용하여 데모 이미지에 대한 추론을 실행합니다.
4.  생성된 텍스트 출력을 실시간으로 스트리밍합니다.

### 코드 이해하기 (Understanding the Code)

`run_qwen2_vl_local.py` 스크립트는 전체 추론 워크플로우를 보여줍니다:

```python
from transformers import TextStreamer
from mblt_model_zoo.transformers import pipeline, AutoModelForImageTextToText, AutoProcessor

# 컴파일된 MXQ 모델 경로 (런타임 디렉토리 기준 상대 경로)
model_folder = "../../../compilation/transformers/vlm/compile/mxq/"
model_id = "mobilint/Qwen2-VL-2B-Instruct"

# 컴파일된 모델 로드
model = AutoModelForImageTextToText.from_pretrained(model_folder)

# Hugging Face에서 프로세서 로드
processor = AutoProcessor.from_pretrained(model_id)

# 파이프라인 생성
pipe = pipeline(
    "image-text-to-text",
    model=model,
    processor=processor,
)

# max_new_tokens 제한 제거
pipe.generation_config.max_new_tokens = None

# 이미지와 함께 메시지 준비
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://..."},
            {"type": "text", "text": "질문 내용을 여기에 입력하세요"},
        ],
    }
]

# 스트리밍으로 추론 실행
pipe(
    text=messages,
    generate_kwargs={
        "max_length": 512,
        "streamer": TextStreamer(tokenizer=pipe.tokenizer, skip_prompt=False),
        "repetition_penalty": 1.1,
    },
)

# 리소스 정리
pipe.model.dispose()
```

### 주요 구성 요소 (Key Components)

#### 1. 모델 로드 (Model Loading)

-   `AutoModelForImageTextToText.from_pretrained()`: 지정된 디렉토리에서 컴파일된 MXQ 모델을 로드합니다.
-   `config.json`을 기반으로 언어 및 비전 모델을 모두 자동으로 감지하고 로드합니다.

#### 2. 프로세서 로드 (Processor Loading)

-   `AutoProcessor.from_pretrained()`: Hugging Face에서 토크나이저와 이미지 프로세서를 로드합니다.
-   프로세서 구성에는 `mobilint/Qwen2-VL-2B-Instruct` 모델 ID를 사용합니다.

#### 3. 파이프라인 생성 (Pipeline Creation)

-   `pipeline()`: `image-text-to-text` 파이프라인을 생성합니다.
-   비전 구성 요소와 언어 구성 요소 간의 상호 작용을 자동으로 처리합니다.

#### 4. 메시지 형식 (Message Format)

-   메시지는 역할(role)과 콘텐츠 유형(content type)이 있는 구조화된 형식을 따릅니다.
-   콘텐츠에는 이미지와 텍스트를 모두 포함할 수 있습니다.
-   이미지는 URL 또는 로컬 파일 경로로 지정할 수 있습니다.

#### 5. 생성 (Generation)

-   `generate_kwargs`: 생성 파라미터(예: `max_length`, `temperature`)를 제어합니다.
-   `TextStreamer`: 생성된 텍스트를 실시간으로 표시합니다.

## 추론 사용자 정의 (Customizing Inference)

### 로컬 이미지 사용 (Using Local Images)

URL 대신 로컬 이미지를 사용하려면:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/path/to/your/image.jpg"},
            {"type": "text", "text": "이 이미지에는 무엇이 있나요?"},
        ],
    }
]
```

### 프롬프트 변경 (Changing the Prompt)

다른 질문을 하려면 텍스트 내용을 수정하십시오:

```python
{"type": "text", "text": "주요 피사체를 둘러싼 환경과 맥락을 설명해 주세요."}
{"type": "text", "text": "이 이미지에서 보이는 물체는 무엇인가요?"}
{"type": "text", "text": "이미지에 있는 사람 수를 세어 주세요."}
{"type": "text", "text": "사물들 간의 공간적 관계는 어떠한가요?"}
```

### 생성 파라미터 조정 (Adjusting Generation Parameters)

다양한 파라미터로 생성 동작을 제어하십시오:

```python
pipe(
    text=messages,
    generate_kwargs={
        "max_length": 1024,      # 더 긴 응답
        "temperature": 0.7,       # 창의성 제어 (높을수록 더 창의적)
        "top_p": 0.9,            # Nucleus 샘플링
        "top_k": 50,             # Top-K 샘플링
        "repetition_penalty": 1.1, # 반복 억제
        "streamer": TextStreamer(tokenizer=pipe.tokenizer, skip_prompt=False),
    },
)
```

### 멀티턴 대화 (Multi-turn Conversations)

이 모델은 멀티턴 대화를 지원합니다:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "image.jpg"},
            {"type": "text", "text": "이 이미지에는 무엇이 있나요?"},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "공원에서 놀고 있는 강아지가 보입니다."}],
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "강아지는 무슨 색인가요?"}],
    },
]
```

## 성능 참고 사항 (Performance Notes)

-   **하드웨어 가속**: 컴파일된 MXQ 모델은 최적의 성능을 위해 Aries 2 하드웨어에서 실행됩니다.
-   **상태 유지 KV 캐시 (Stateful KV Cache)**: 상태 유지 KV 캐시 시스템을 통해 메모리가 효율적으로 관리됩니다.
-   **스트리밍 출력**: 더 나은 사용자 경험을 위해 텍스트 생성이 실시간으로 스트리밍됩니다.
-   **리소스 정리**: 추론 후 리소스를 올바르게 정리하려면 항상 `pipe.model.dispose()`를 호출하십시오.

## 구성 (Configuration)

### 모델 경로 (Model Path)

스크립트는 컴파일된 모델에 대한 상대 경로를 사용합니다:

```python
model_folder = "../../../compilation/transformers/vlm/compile/mxq/"
```

이 상대 경로는 `runtime/transformers/vlm/` 디렉토리에서 스크립트를 실행할 때 작동합니다. 모델을 다른 위치에 컴파일한 경우 이 경로를 그에 맞게 업데이트하십시오.

### 모델 ID (Model ID)

프로세서는 모델 ID를 사용하여 Hugging Face에서 로드됩니다:

```python
model_id = "mobilint/Qwen2-VL-2B-Instruct"
```

이 ID는 토크나이저 및 이미지 프로세서 구성을 다운로드하는 데 사용됩니다.

## 문제 해결 (Troubleshooting)

### 모델을 찾을 수 없음 오류 (Model Not Found Error)

```text
FileNotFoundError: Model files not found
```

**해결 방법**: 컴파일 튜토리얼을 완료했고 컴파일 출력 디렉토리에 4개의 파일이 모두 존재하는지 확인하십시오.

### 메모리 부족 (OOM) 오류

**해결 방법**:

-   생성 파라미터에서 `max_length`를 줄이십시오.
-   더 작은 이미지를 처리하십시오.
-   GPU/NPU를 많이 사용하는 다른 애플리케이션을 종료하십시오.

### 프로세서 다운로드 문제

**해결 방법**:

-   인터넷 연결을 확인하십시오.
-   Hugging Face 액세스 권한을 확인하십시오 (인증이 필요할 수 있음).
-   필요한 경우 `huggingface-cli login`을 사용하십시오.

### 가져오기 오류 (Import Errors)

```text
ModuleNotFoundError: No module named 'mblt_model_zoo'
```

**해결 방법**: 필수 패키지를 설치하십시오:

```bash
pip install mblt-model-zoo
```

## 출력 예시 (Example Outputs)

### 이미지 설명

**입력**: "주요 피사체를 둘러싼 환경과 맥락을 설명해 주세요."
**출력**: 장면, 사물, 공간적 관계에 대한 자세한 설명.

### 객체 수 세기

**입력**: "이 이미지에 사람이 몇 명 있나요?"
**출력**: 보이는 사람의 수와 설명.

### 시각적 추론

**입력**: "이 사람은 무엇을 하고 있나요?"
**출력**: 이미지 속 행동 및 활동 분석.

### 공간 이해

**입력**: "강아지는 나무와 비교했을 때 어디에 있나요?"
**출력**: 사물 간의 공간적 관계 설명.

## 참고 문헌 (References)

-   [컴파일 튜토리얼](../../../compilation/transformers/vlm/README.md)
-   [Qwen2-VL 모델 카드](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
-   [Mobilint 문서](https://docs.mobilint.com)
-   [mblt-model-zoo 문서](https://docs.mobilint.com/model-zoo)

## 지원 (Support)

문제나 질문이 있는 경우:

-   위의 문제 해결 섹션을 확인하십시오.
-   모델이 제대로 컴파일되었는지 확인하려면 컴파일 튜토리얼을 검토하십시오.
-   `mblt-model-zoo` 문서를 참조하십시오.
-   상세한 오류 로그와 함께 Mobilint 지원팀에 문의하십시오.

> **참고**: 이 런타임 추론 스크립트를 사용하려면 컴파일 튜토리얼에서 올바르게 컴파일된 MXQ 모델이 필요합니다. 추론을 실행하기 전에 컴파일 과정을 완료했는지 확인하십시오.
