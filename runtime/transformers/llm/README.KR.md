# 대규모 언어 모델 런타임 가이드

이 튜토리얼은 Mobilint QB 런타임과 maccel 가속기 라이브러리를 사용하여 컴파일된 대규모 언어 모델로 추론을 실행하는 방법에 대한 상세한 지침을 제공합니다.

이 가이드는 `mblt-sdk-tutorial/compilation/transformers/llm/README.md`에서 이어집니다. 모델을 성공적으로 컴파일했으며 다음 파일들이 준비되어 있다고 가정합니다:

- `Llama-3.2-1B-Instruct.mxq` - 컴파일된 모델 파일
- `embedding.pt` - 임베딩 레이어 가중치

## 사전 요구사항

추론을 실행하기 전에 다음이 준비되어 있는지 확인하세요:

- maccel 런타임 라이브러리 (NPU 가속기 접근 제공)
- 컴파일된 `.mxq` 모델 파일
- 임베딩 가중치 파일

## 개요

추론 프로세스는 사용자 정의 `LlamaMXQ` 모델 클래스를 사용하며, 이 클래스는:

1. maccel 가속기를 통해 컴파일된 `.mxq` 모델을 Mobilint NPU에 로드합니다
2. 토큰-벡터 변환을 위해 CPU 기반 임베딩 레이어를 사용합니다
3. NPU 가속화된 트랜스포머 레이어를 통해 프롬프트를 처리합니다
4. 표준 HuggingFace 생성 유틸리티를 사용하여 텍스트를 생성합니다

---

## 추론 실행

예제 추론 스크립트를 실행하려면:

```bash
python inference_mxq.py --mxq_path ../../../compilation/transformers/llm/Llama-3.2-1B-Instruct.mxq --embedding_weight_path ../../../compilation/transformers/llm/embedding.pt
```

**이 스크립트의 동작:**

- HuggingFace에서 토크나이저를 로드합니다
- 컴파일된 `.mxq` 파일로 `LlamaMXQ` 모델을 초기화합니다
- NPU 가속화된 모델을 사용하여 응답을 생성합니다
- 생성된 텍스트 출력을 표시합니다

**중요한 설정:**

- `MODEL_NAME`을 적절한 모델 ID로 설정하세요
- 디바이스는 `'cpu'`로 설정됩니다 - 모델이 NPU에서 실행되므로 GPU를 사용하지 마세요
