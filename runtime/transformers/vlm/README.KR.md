# 비전 언어 모델 (VLM) 런타임 추론

이 디렉토리는 Mobilint 런타임을 사용하여 Aries 2 하드웨어에서 컴파일된 Qwen2-VL-2B-Instruct 모델을 실행하기 위한 런타임 추론 코드를 포함합니다.

## 개요

컴파일 튜토리얼(`/workspace/mblt-sdk-tutorial/compilation/transformers/vlm/`)을 사용하여 VLM 모델을 컴파일한 후, 이 런타임 스크립트를 사용하여 컴파일된 MXQ 모델로 추론을 실행할 수 있습니다.

추론 스크립트는 다음을 수행하는 방법을 보여줍니다:

- 컴파일 출력 디렉토리에서 컴파일된 MXQ 모델 로드
- 이미지-텍스트-텍스트 작업을 위한 Mobilint 런타임 파이프라인 사용
- 스트리밍 출력으로 추론 실행
- 이미지가 포함된 비전-언어 쿼리 처리

## 사전 요구사항

추론을 실행하기 전에 다음이 준비되어 있는지 확인하세요:

- **컴파일 튜토리얼 완료** - 컴파일 디렉토리에 4개의 파일이 모두 생성되어 있어야 합니다
- **mblt-model-zoo 패키지 설치**:

  ```bash
  pip install mblt-model-zoo
  ```

- **필수 의존성 패키지 설치**:

  ```bash
  pip install transformers==4.54.0 torch pillow
  ```

## 필수 파일

추론 스크립트는 컴파일 출력 디렉토리(런타임 디렉토리 기준 `../../../compilation/transformers/vlm/compile/mxq/`)에 다음 4개의 파일이 존재할 것으로 예상합니다:

1. **Qwen2-VL-2B-Instruct_text_model.mxq** - 컴파일된 언어 모델
2. **Qwen2-VL-2B-Instruct_vision_transformer.mxq** - 컴파일된 비전 인코더
3. **config.json** - MXQ 경로가 포함된 모델 구성
4. **model.safetensors** - 회전된 임베딩 가중치

이러한 파일들은 컴파일 튜토리얼을 따라하면 자동으로 생성됩니다.

## 추론 실행

### 기본 사용법

추론 스크립트를 실행하기만 하면 됩니다:

```bash
cd /workspace/mblt-sdk-tutorial/runtime/transformers/vlm
python run_qwen2_vl_local.py
```

스크립트는 다음을 수행합니다:

1. 컴파일 출력 디렉토리에서 컴파일된 MXQ 모델을 로드합니다
2. HuggingFace에서 프로세서를 로드합니다 (모델 ID: `mobilint/Qwen2-VL-2B-Instruct`)
3. 데모 이미지에 샘플 프롬프트로 추론을 실행합니다
4. 생성된 텍스트 출력을 실시간으로 스트리밍합니다

### 코드 이해

`run_qwen2_vl_local.py` 스크립트는 완전한 추론 워크플로우를 보여줍니다:

```python
from transformers import TextStreamer
from mblt_model_zoo.transformers import pipeline, AutoModelForImageTextToText, AutoProcessor

# 컴파일된 MXQ 모델 경로 (런타임 디렉토리 기준 상대 경로)
model_folder = "../../../compilation/transformers/vlm/compile/mxq/"
model_id = "mobilint/Qwen2-VL-2B-Instruct"

# 컴파일된 모델 로드
model = AutoModelForImageTextToText.from_pretrained(model_folder)

# HuggingFace에서 프로세서 로드
processor = AutoProcessor.from_pretrained(model_id)

# 파이프라인 생성
pipe = pipeline(
    "image-text-to-text",
    model=model,
    processor=processor,
)

# max_new_tokens 제한 해제
pipe.generation_config.max_new_tokens = None

# 이미지가 포함된 메시지 준비
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://..."},
            {"type": "text", "text": "Your question here"},
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

# 정리
pipe.model.dispose()
```

### 주요 구성 요소

#### 1. 모델 로딩

- **AutoModelForImageTextToText.from_pretrained()** - 지정된 디렉토리에서 컴파일된 MXQ 모델을 로드합니다
- config.json을 기반으로 언어 모델과 비전 모델을 자동으로 감지하고 로드합니다

#### 2. 프로세서 로딩

- **AutoProcessor.from_pretrained()** - HuggingFace에서 토크나이저와 이미지 프로세서를 로드합니다
- 프로세서 구성을 위해 모델 ID `mobilint/Qwen2-VL-2B-Instruct`를 사용합니다

#### 3. 파이프라인 생성

- **pipeline()** - 이미지-텍스트-텍스트 파이프라인을 생성합니다
- 비전 및 언어 구성 요소 간의 상호 작용을 자동으로 처리합니다

#### 4. 메시지 형식

- 메시지는 역할과 콘텐츠 유형이 있는 구조화된 형식을 사용합니다
- 콘텐츠에는 이미지와 텍스트가 모두 포함될 수 있습니다
- 이미지는 URL 또는 로컬 파일 경로로 지정할 수 있습니다

#### 5. 생성

- **generate_kwargs** - 생성 매개변수를 제어합니다
- **TextStreamer** - 생성된 텍스트를 실시간으로 표시합니다
- **max_length** - 최대 출력 길이를 제어합니다

## 추론 사용자 정의

### 로컬 이미지 사용

URL 대신 로컬 이미지를 사용하려면:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/path/to/your/image.jpg"},
            {"type": "text", "text": "What is in this image?"},
        ],
    }
]
```

### 프롬프트 변경

다른 질문을 하기 위해 텍스트 콘텐츠를 수정합니다:

```python
{"type": "text", "text": "Describe the environment and context surrounding the main subject."}
{"type": "text", "text": "What objects are visible in this image?"}
{"type": "text", "text": "Count the number of people in the image."}
{"type": "text", "text": "What is the spatial relationship between objects?"}
```

### 생성 매개변수 조정

다른 매개변수로 생성 동작을 제어합니다:

```python
pipe(
    text=messages,
    generate_kwargs={
        "max_length": 1024,      # 더 긴 응답
        "temperature": 0.7,       # 창의성 제어 (높을수록 더 창의적)
        "top_p": 0.9,            # Nucleus 샘플링
        "top_k": 50,             # Top-K 샘플링
        "repetition_penalty": 1.1, # 반복 패널티
        "streamer": TextStreamer(tokenizer=pipe.tokenizer, skip_prompt=False),
    },
)
```

### 다중 턴 대화

모델은 다중 턴 대화를 지원합니다:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "image.jpg"},
            {"type": "text", "text": "What's in this image?"},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "I see a dog playing in a park."}],
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "What color is the dog?"}],
    },
]
```

## 성능 참고사항

- **하드웨어 가속**: 컴파일된 MXQ 모델은 최적의 성능을 위해 Aries 2 하드웨어에서 실행됩니다
- **상태 저장 KV 캐시**: 메모리는 상태 저장 KV 캐시 시스템을 통해 효율적으로 관리됩니다
- **스트리밍 출력**: 텍스트 생성은 더 나은 사용자 경험을 위해 실시간으로 스트리밍됩니다
- **리소스 정리**: 추론 후 항상 `pipe.model.dispose()`를 호출하여 리소스를 적절히 정리하세요

## 구성

### 모델 경로

스크립트는 컴파일된 모델에 대한 상대 경로를 사용합니다:

```python
model_folder = "../../../compilation/transformers/vlm/compile/mxq/"
```

이 상대 경로는 `runtime/transformers/vlm/` 디렉토리에서 스크립트를 실행할 때 작동합니다. 모델을 다른 위치에 컴파일한 경우, 이 경로를 그에 맞게 업데이트하세요.

### 모델 ID

프로세서는 모델 ID를 사용하여 HuggingFace에서 로드됩니다:

```python
model_id = "mobilint/Qwen2-VL-2B-Instruct"
```

이 ID는 토크나이저와 이미지 프로세서 구성을 다운로드하는 데 사용됩니다.

## 문제 해결

### 모델을 찾을 수 없음 오류

```text
FileNotFoundError: Model files not found
```

**해결 방법**: 컴파일 튜토리얼을 완료했고 컴파일 출력 디렉토리에 4개의 파일이 모두 존재하는지 확인하세요.

### 메모리 부족 (OOM) 오류

**해결 방법**:

- 생성 매개변수에서 `max_length` 줄이기
- 더 작은 이미지 처리
- 다른 GPU 집약적 애플리케이션 닫기

### 프로세서 다운로드 문제

**해결 방법**:

- 인터넷 연결 확인
- HuggingFace 접근 확인 (인증이 필요할 수 있음)
- 필요시 `huggingface-cli login` 사용

### 가져오기 오류

```text
ModuleNotFoundError: No module named 'mblt_model_zoo'
```

**해결 방법**: 필수 패키지 설치:

```bash
pip install mblt-model-zoo
```

## 예제 출력

### 이미지 설명

**입력**: "Describe the environment and context surrounding the main subject."
**출력**: 장면, 객체 및 공간 관계에 대한 상세한 설명

### 객체 카운팅

**입력**: "How many people are in this image?"
**출력**: 보이는 사람들의 수와 설명

### 시각적 추론

**입력**: "What is the person doing?"
**출력**: 이미지의 동작 및 활동 분석

### 공간 이해

**입력**: "Where is the dog located relative to the tree?"
**출력**: 객체 간 공간 관계 설명

## 참고 자료

- [컴파일 튜토리얼](../../../compilation/transformers/vlm/README.md)
- [Qwen2-VL 모델 카드](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [Mobilint 문서](https://docs.mobilint.com)
- [mblt-model-zoo 문서](https://docs.mobilint.com/model-zoo)

## 지원

문제나 질문이 있으시면:

- 위의 문제 해결 섹션 확인
- 모델이 제대로 컴파일되었는지 컴파일 튜토리얼 검토
- mblt-model-zoo 문서 참조
- 상세한 오류 로그와 함께 Mobilint 지원팀에 문의

> **참고**: 이 런타임 추론 스크립트는 컴파일 튜토리얼에서 제대로 컴파일된 MXQ 모델이 필요합니다. 추론을 실행하기 전에 컴파일 프로세스를 완료했는지 확인하세요.
