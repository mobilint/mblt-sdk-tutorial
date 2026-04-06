# 런타임 파이프라인 개요

이 문서는 컴파일된 `.mxq` 모델을 Mobilint NPU에서 추론하기까지의 전체 흐름을 설명합니다.

> 컴파일 과정에 대해서는
> [컴파일 파이프라인 개요](../../compilation/_guides/00_about_compilation_pipeline.KR.md)를 참조하세요.

---

## 런타임 파이프라인

![Runtime Pipeline](../../assets/runtime_pipeline.png)

---

## 추론 API

Mobilint NPU에서 모델을 추론하는 방법은 두 가지입니다:

- **`qbruntime`** (low-level) — NPU를 직접 제어. `.mxq` 파일을 로드하고 numpy 배열로 추론
- **[`mblt-model-zoo`](https://github.com/mobilint/mblt-model-zoo)** (high-level) — HuggingFace 호환 API. 임베딩 분리, KV cache 관리, 토큰 생성 루프 등을 자동 처리

inference 파이프라인을 직접 구성하기 까다로운 모델(LLM, VLM, STT 등)은
`mblt-model-zoo` 사용을 권장합니다.

> 자세한 내용은 [추론 API 가이드](./02_about_inference_api.KR.md)를 참조하세요.

---

## 1단계: 모델 준비

`qbruntime`을 사용하는 경우 `.mxq` 파일만으로 바로 추론이 가능하여 별도의 준비 단계가 필요 없습니다.

`mblt-model-zoo`를 사용하는 경우 `prepare_model.py`를 실행하여 모델 폴더를 구성해야 합니다.

이 스크립트가 수행하는 작업:

1. `.mxq` 파일을 출력 폴더로 복사
2. 임베딩 가중치를 safetensors 형식으로 변환
3. HuggingFace에서 config.json, tokenizer 등을 다운로드
4. config.json에 `mxq_path`, `_name_or_path`, `target_cores` 등 NPU 설정 추가

준비된 모델 폴더 구조:

```text
model-folder/
├── config.json              # mxq_path, target_cores 등 NPU 설정 포함
├── model.safetensors        # 임베딩 가중치
├── {model}.mxq              # 컴파일된 모델 (1개 이상)
├── tokenizer.json           # 토크나이저 (텍스트 모델의 경우)
└── generation_config.json   # 생성 설정 (선택)
```

> 자세한 내용은 [모델 준비 가이드](./01_about_model_preparation.KR.md)를 참조하세요.

---

## 2단계: 추론 실행

선택한 API(`qbruntime` 또는 `mblt-model-zoo`)로 추론을 실행합니다.

> 각 API의 사용법은 [추론 API 가이드](./02_about_inference_api.KR.md)를 참조하세요.

---

## 3단계: 리소스 정리

추론이 끝나면 반드시 NPU 리소스를 해제해야 합니다.
NPU는 공유 리소스이므로, 정리하지 않으면 다른 프로세스가 NPU에 접근할 수 없습니다.

```python
# qbruntime 사용 시
mxq_model.dispose()

# mblt-model-zoo 사용 시
model.dispose()

# 멀티 컴포넌트 모델의 경우 각 서브모델을 개별 정리
pipe.model.model.visual.dispose()
pipe.model.model.language_model.dispose()
```

---

## 다음 문서

- [모델 준비 가이드](./01_about_model_preparation.KR.md) - prepare_model.py와 config.json 구조
- [추론 API 가이드](./02_about_inference_api.KR.md) - qbruntime vs mblt-model-zoo
