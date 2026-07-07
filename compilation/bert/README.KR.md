# Bidirectional Encoder Representations from Transformers (BERT)

이 튜토리얼에서는 Mobilint qb Compiler로 BERT 모델을 컴파일하는 방법을 안내합니다. 이 과정은 일반적인 BERT 모델을 Mobilint NPU 하드웨어에서 효율적으로 실행할 수 있는 최적화된 `.mxq` 파일로 변환합니다.

이 튜토리얼에서는 문장 임베딩 생성을 위해 조정된 BERT 기반 모델인 [Sentence-BERT](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors)를 사용합니다.

## 개요

전체 워크플로는 네 단계로 구성됩니다:

1. **임베딩 가중치 추출**: 지원되지 않는 임베딩 레이어를 CPU 측 가중치로 추출
2. **캘리브레이션 데이터 생성**: 양자화에 사용할 캘리브레이션 데이터 생성
3. **MBLT 컴파일**: 모델을 MBLT (Mobilint Binary LayouT) 형식으로 컴파일
4. **MXQ 컴파일**: 양자화를 적용해 최종 `.mxq` 파일 생성

모든 스크립트는 `bert/` 디렉토리에서 실행합니다.

## 사전 준비

- Mobilint qb Compiler (`>= 1.0.0`)
- CUDA 지원 GPU (컴파일 시간 단축을 위해 권장)

```bash
pip install -r requirements.txt
```

## 1단계: 임베딩 가중치 추출

BERT 아키텍처 특성상 일부 입력 임베딩 레이어는 NPU에서 지원되지 않습니다. 이 단계에서는 해당 임베딩 가중치를 모델에서 추출해 CPU 측에서 사용할 수 있도록 `.pth` 파일로 저장합니다.

```bash
python get_embedding.py
```

**실행 내용:**

- Hugging Face에서 Sentence-BERT 모델 로드
- 단어, 토큰 타입, 위치 임베딩과 LayerNorm 가중치 추출
- 가중치 딕셔너리로 저장

**출력:**

- `./weights/weight_dict.pth` - 추출된 임베딩 가중치

> **팁:** 3단계 이후 [Netron](https://netron.mobilint.com)에서 컴파일된 모델을 열어 보면, 어떤 레이어가 NPU에서 실행되고 어떤 레이어가 CPU로 오프로드되는지 확인할 수 있습니다.

## 2단계: 캘리브레이션 데이터 생성

[STS Benchmark Dataset](https://huggingface.co/datasets/mteb/stsbenchmark-sts)을 사용해 캘리브레이션 데이터를 생성합니다. 이 데이터는 MXQ 컴파일 시 양자화에 필요합니다.

```bash
python prepare_calib.py
```

**실행 내용:**

- STS Benchmark 검증 세트에서 문장 로드
- 1단계에서 추출한 임베딩 가중치를 사용해 토큰화 및 임베딩
- 임베딩된 텍스트를 캘리브레이션용 NumPy 파일로 저장

**출력:**

- `./calibration_data/` - 캘리브레이션 `.npy` 파일이 포함된 디렉토리

## 3단계: MBLT 컴파일

BERT 모델을 중간 형식인 MBLT (Mobilint Binary LayouT)로 컴파일합니다.

```bash
# ARIES (기본값)
python compile_mblt.py

# REGULUS (2026-06 이후 고객)
python compile_mblt.py --target-device regulus-rb
```

`compile_mblt.py`는 `mblt_compile()`를 호출하며, `--target-device`로 대상 NPU를 선택합니다 (기본값: `aries-rb`).

**실행 내용:**

- Hugging Face에서 Sentence-BERT 모델 로드
- 시퀀스 길이 차원을 동적으로 설정
- 어텐션 마스크를 패딩 마스크로 설정
- 미지원 레이어의 CPU 오프로드와 함께 MBLT 형식으로 컴파일

**출력:**

- `./mblt/stsb-bert-tiny-safetensors.mblt` - 중간 MBLT 형식

## 4단계: MXQ 컴파일

2단계에서 생성한 캘리브레이션 데이터를 사용해 모델을 최종 `.mxq` 형식으로 양자화 컴파일합니다.

```bash
# ARIES (기본값)
python compile_mxq.py

# REGULUS (2026-06 이후 고객)
python compile_mxq.py --target-device regulus-rb
```

`compile_mxq.py`는 `--target-device`로 대상 NPU를 선택합니다 (기본값: `aries-rb`). REGULUS는 `inference_scheme="single"`만 지원하므로 `regulus` 디바이스를 선택하면 자동으로 설정됩니다.

**실행 내용:**

- Hugging Face에서 Sentence-BERT 모델 로드
- MaxPercentile 양자화를 포함한 `CalibrationConfig` 적용:
  - 방법: WChAMulti (가중치 채널별, 활성화 다중 레이어)
  - 출력: 레이어별 양자화
  - Percentile: 0.999, Top-k 비율: 0.01
- 2단계의 캘리브레이션 데이터를 사용해 `.mxq` 형식으로 컴파일

**출력:**

- `./mxq/stsb-bert-tiny-safetensors.mxq` - NPU용 최종 양자화 모델

### 대상 디바이스 (`--target-device`)

| 사용자 | `--target-device` |
| --- | --- |
| ARIES | `aries-rb` (기본값) |
| REGULUS (2026-06 이후 고객) | `regulus-rb` |

> **참고:** BERT 컴파일은 신형 REGULUS(`regulus-rb`, 2026-06 이후 고객)에서 지원됩니다. 구형 REGULUS(`regulus-ra`, 2026-06 이전 고객)는 이 작업을 지원하지 않습니다. `compile_mblt.py`와 `compile_mxq.py` 모두에 `--target-device regulus-rb`를 사용하세요.

## 파일 구조

```text
bert/
├── get_embedding.py
├── prepare_calib.py
├── compile_mblt.py
├── compile_mxq.py
├── requirements.txt
├── README.md
├── README.KR.md
├── weights/                               # 추출된 임베딩 가중치
│   └── weight_dict.pth
├── calibration_data/                      # 캘리브레이션 데이터
│   └── *.npy
├── mblt/                                  # 중간 MBLT 모델
│   └── stsb-bert-tiny-safetensors.mblt
└── mxq/                                   # 출력 MXQ 모델
    └── stsb-bert-tiny-safetensors.mxq
```

## 문제 해결

### 임베딩 가중치 누락

캘리브레이션 중 임베딩 가중치가 없다는 오류가 발생하면 다음을 확인하세요:

```bash
ls ./weights/weight_dict.pth
```

파일이 없으면 `get_embedding.py`를 다시 실행하세요.

### 캘리브레이션 데이터 누락

MXQ 컴파일 중 캘리브레이션 데이터가 없다는 오류가 발생하면 다음을 확인하세요:

```bash
ls ./calibration_data/
```

디렉토리가 없거나 비어 있으면 `prepare_calib.py`를 다시 실행하세요.

## 참고 자료

- [Sentence-BERT](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors)
- [STS Benchmark 데이터셋](https://huggingface.co/datasets/mteb/stsbenchmark-sts)
- [Mobilint 문서](https://docs.mobilint.com)
