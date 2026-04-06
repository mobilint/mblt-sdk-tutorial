# 멀티 컴포넌트 모델 가이드

VLM(Vision-Language Model)이나 STT(Speech-to-Text)처럼 여러 서브모델로 구성된 모델은
각 컴포넌트를 **개별적으로 컴파일**해야 합니다.

이 문서는 분리 컴파일이 필요한 경우와 주의사항을 설명합니다.

---

## 왜 분리 컴파일이 필요한가?

단일 모델(예: ResNet, BERT)은 하나의 `.mxq` 파일로 컴파일됩니다.

하지만 여러 서브모델로 구성된 모델은 아키텍처 특성상 분리가 필요합니다:

- **서브모델 간 입력 형태가 다른 경우** — 예: VLM에서 vision encoder는 이미지, language model은 텍스트 임베딩을 입력으로 받음
- **서브모델 간 추론 호출 횟수가 다른 경우** — 예: STT에서 encoder는 1회 호출, decoder는 토큰 수만큼 반복 호출
- **서브모델별 양자화 설정이 다른 경우** — 예: autoregressive decoder는 `LlmConfig`가 필요하지만 encoder는 불필요

---

## 컴파일 순서 의존성

서브모델 간에 컴파일 순서 의존성이 있을 수 있습니다.

### 순서 의존성이 있는 경우 (VLM)

SpinQuant의 R1을 사용하는 경우,
language model 컴파일 시 생성된 R1 회전 행렬을 vision encoder 컴파일에서 참조해야 합니다.

```text
[1] Language Model 컴파일  ──→  R1 회전 행렬 생성 (spinWeight/)
                                       │
[2] Vision Encoder 컴파일  ←──  R1 참조 (HeadOutChRotation)
```

vision encoder의 출력이 회전된 language model의 입력 공간과 정렬되어야 하기 때문에,
**language model을 먼저 컴파일해야 합니다**.

> 자세한 구현은 `vlm/` 튜토리얼을 참조하세요.

### 순서 의존성이 없는 경우 (STT)

encoder와 decoder 간에 가중치 공간 정렬이 필요 없다면,
각 컴포넌트를 독립적으로 컴파일할 수 있습니다.

```text
[1] Encoder 컴파일  (독립)
[2] Decoder 컴파일  (독립)
```

단, decoder가 autoregressive 구조인 경우 `LlmConfig`를 전달해야 합니다.
encoder는 고정 길이 입력을 받으므로 `LlmConfig`가 불필요합니다.

> 자세한 구현은 `stt/` 튜토리얼을 참조하세요.

---

## 임베딩 레이어 분리

VLM, LLM, BERT 등에서 **임베딩 레이어는 NPU에서 실행되지 않습니다**.

따라서:

1. 임베딩 가중치를 별도로 추출
2. CPU에서 임베딩 룩업 수행
3. 결과를 NPU 모델에 입력

SpinQuant(R1)를 사용하는 경우, 추출한 임베딩에 R1 회전을 적용해야 합니다.

> 임베딩 회전에 대한 자세한 내용은
> [컴파일 Config 가이드 - SpinQuant 상세 설명](./01_about_quantization_config.KR.md#spinquant-r1r2-상세-설명)을 참조하세요.

---

## 최종 배포 패키지

멀티 컴포넌트 모델의 컴파일 결과는 여러 `.mxq` 파일과 설정 파일로 구성됩니다.

```text
mxq/
├── {component_1}.mxq          # 서브모델 1
├── {component_2}.mxq          # 서브모델 2
├── config.json                # mxq_path 등이 추가된 설정 파일
└── model.safetensors          # 임베딩 가중치 (필요 시 회전 적용됨)
```

`config.json`에는 각 서브모델의 `.mxq` 경로(`mxq_path`)가 추가되어
runtime에서 올바른 모델 파일을 로드할 수 있도록 합니다.

---

## 다음 문서

- [컴파일 파이프라인 개요](./00_about_compilation_pipeline.KR.md) - 전체 컴파일 흐름
- [컴파일 Config 가이드](./01_about_quantization_config.KR.md) - 컴포넌트별 양자화 설정 차이
- [Calibration 데이터 가이드](./02_about_calibration_data.KR.md) - 컴포넌트별 calibration 데이터 구조
