# 대형 언어 모델 추론 (Large Language Model Inference)

이 튜토리얼은 Mobilint qbruntime을 사용하여 컴파일된 대형 언어 모델(LLM)로 추론을 실행하는 방법에 대한 단계별 지침을 제공합니다.

이 가이드는 [mblt-sdk-tutorial/compilation/transformers/llm/README.md](file:///workspace/mblt-sdk-tutorial/compilation/transformers/llm/README.md)에서 이어지는 내용입니다. 모델 컴파일을 성공적으로 마쳤으며 다음 파일이 준비되어 있다고 가정합니다:

- `./Llama-3.2-1B-Instruct.mxq` - 컴파일된 모델 파일
- `./embedding.pt` - 임베딩 레이어 가중치 (PyTorch 포맷)

## 사전 요구 사항 (Prerequisites)

추론을 실행하기 전에 다음 구성 요소가 설치되어 있고 준비되었는지 확인하십시오:

- `qbruntime` 라이브러리 (NPU 가속기 액세스용)
- 컴파일된 `.mxq` 모델 파일
- 임베딩 가중치 파일 (`.pt`)
- Python 패키지: `torch`, `transformers`

## 개요 (Overview)

추론 로직은 NPU 가속기와 Hugging Face 생태계를 통합하는 맞춤형 `LlamaMXQ` 모델 클래스를 사용합니다. 워크플로우는 다음과 같습니다:

1.  **초기화 (Initialization)**: `qbruntime` 가속기를 통해 컴파일된 `.mxq` 모델을 Mobilint NPU에 로드합니다.
2.  **임베딩 (Embedding)**: CPU 기반 임베딩 레이어를 사용하여 입력 토큰을 벡터로 변환합니다.
3.  **추론 (Inference)**: NPU 가속 트랜스포머 레이어를 통해 프롬프트를 처리합니다.
4.  **생성 (Generation)**: 표준 Hugging Face 생성 유틸리티를 사용하여 텍스트를 생성합니다.

## 추론 실행 (Running Inference)

예제 추론 스크립트를 실행하려면 다음 명령어를 사용하십시오:

```bash
python inference_mxq.py --mxq_path ../../../compilation/transformers/llm/Llama-3.2-1B-Instruct.mxq --embedding_weight_path ../../../compilation/transformers/llm/embedding.pt
```

### 스크립트 세부 설명

- **토크나이저 로드 (Tokenizer Loading)**: 텍스트 입력을 처리하기 위해 Hugging Face에서 토크나이저를 로드합니다.
- **모델 초기화 (Model Initialization)**: 컴파일된 `.mxq` 파일로 `LlamaMXQ` 모델을 초기화합니다.
- **생성 (Generation)**: NPU 가속 모델을 사용하여 응답을 생성합니다.
- **출력 (Output)**: 생성된 텍스트 출력을 표시합니다.

### 파라미터 (Parameters)

- `--mxq_path`: 컴파일된 `.mxq` 모델 파일의 경로입니다.
- `--embedding_weight_path`: 임베딩 가중치 파일(`.pt`)의 경로입니다.
- **참고**: 모델이 내부적으로 많은 계산을 NPU로 오프로드하므로 스크립트에서 장치는 명시적으로 `'cpu'`로 설정됩니다. GPU를 사용하지 **마십시오**.

### 예상 출력 (Expected Output)

스크립트는 입력 프롬프트를 기반으로 생성된 텍스트 응답을 출력합니다.
