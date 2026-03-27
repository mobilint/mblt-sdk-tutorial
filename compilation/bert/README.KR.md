# Bidirectional Encoder Representations from Transformers (BERT)

이 튜토리얼은 Mobilint qb 컴파일러를 사용하여 BERT 모델을 컴파일하는 세부 지침을 제공합니다. 컴파일 프로세스는 표준 BERT 모델을 Mobilint NPU 하드웨어에서 효율적으로 실행할 수 있는 최적화된 `.mxq` 형식으로 변환합니다.

이 튜토리얼에서는 BERT 아키텍처를 기반으로 하고 문장 임베딩을 생성하도록 수정된 [Sentence-BERT](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) 모델을 사용합니다.

## 개요

컴파일 과정은 네 가지 주요 단계로 구성됩니다:

1. **임베딩 가중치 추출**: 미지원 임베딩 레이어를 CPU 측 가중치로 추출
2. **캘리브레이션 데이터 생성**: 양자화를 위한 캘리브레이션 데이터셋 생성
3. **MBLT 컴파일**: 모델을 MBLT (Mobilint Binary LayouT) 형식으로 컴파일
4. **MXQ 컴파일**: 양자화를 적용하여 `.mxq` 형식으로 컴파일

모든 스크립트는 `bert/` 디렉토리에서 실행합니다.

## 사전 준비

- Mobilint qb 컴파일러 (버전 1.0.0 이상 필요)
- CUDA 지원 GPU (컴파일 시간 단축을 위해 권장)

```bash
pip install -r requirements.txt
```

## 1단계: 임베딩 가중치 추출

BERT의 복잡한 아키텍처로 인해 일부 입력 임베딩 레이어는 NPU에서 지원되지 않습니다. 따라서 모델에서 임베딩 가중치를 추출하여 CPU 측에서 실행할 수 있도록 `.pth` 파일로 저장합니다.

```bash
python get_embedding.py
```

**실행 내용:**

- HuggingFace에서 Sentence-BERT 모델 로드
- 단어, 토큰 타입, 위치 임베딩 및 LayerNorm 가중치 추출
- 가중치 딕셔너리로 저장

**출력:**

- `./weights/weight_dict.pth` - 추출된 임베딩 가중치

> **팁:** MBLT 컴파일(3단계) 후 [Netron](https://netron.mobilint.com)을 사용하여 모델 아키텍처를 시각화하면 어떤 레이어가 지원되고 어떤 레이어가 CPU로 오프로드되는지 확인할 수 있습니다.

## 2단계: 캘리브레이션 데이터 생성

[STS Benchmark Dataset](https://huggingface.co/datasets/mteb/stsbenchmark-sts)을 사용하여 캘리브레이션 데이터를 생성합니다. 이 데이터는 MXQ 컴파일 시 양자화에 필수적입니다.

```bash
python prepare_calib.py
```

**실행 내용:**

- STS Benchmark 검증 세트에서 문장 로드
- 추출된 임베딩 가중치(1단계)를 사용하여 토큰화 및 임베딩
- 임베딩된 텍스트를 캘리브레이션용 NumPy 파일로 저장

**출력:**

- `./calibration_data/` - 캘리브레이션 `.npy` 파일이 포함된 디렉토리

## 3단계: MBLT 컴파일

BERT 모델을 MBLT (Mobilint Binary LayouT) 중간 형식으로 컴파일합니다.

```bash
python compile_mblt.py
```

**실행 내용:**

- HuggingFace에서 Sentence-BERT 모델 로드
- 시퀀스 길이 차원을 동적으로 설정
- 어텐션 마스크를 패딩 마스크로 구성
- 미지원 레이어의 CPU 오프로드와 함께 MBLT 형식으로 컴파일

**출력:**

- `./mblt/stsb-bert-tiny-safetensors.mblt` - 중간 MBLT 형식

## 4단계: MXQ 컴파일

캘리브레이션 데이터를 사용하여 최종 `.mxq` 형식으로 양자화 컴파일합니다.

```bash
python compile_mxq.py
```

**실행 내용:**

- HuggingFace에서 Sentence-BERT 모델 로드
- MaxPercentile 양자화를 포함한 `CalibrationConfig` 적용:
  - 방법: WChAMulti (가중치 채널별, 활성화 다중 레이어)
  - 출력: 레이어별 양자화
  - Percentile: 0.999, Top-k 비율: 0.01
- 2단계의 캘리브레이션 데이터를 사용하여 `.mxq` 형식으로 컴파일

**출력:**

- `./mxq/stsb-bert-tiny-safetensors.mxq` - NPU용 최종 양자화 모델

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

캘리브레이션 시 가중치 누락 오류가 발생하는 경우:

```bash
ls ./weights/weight_dict.pth
```

파일이 없으면 `get_embedding.py`를 다시 실행하세요.

### 캘리브레이션 데이터 누락

MXQ 컴파일 시 캘리브레이션 데이터 누락 오류가 발생하는 경우:

```bash
ls ./calibration_data/
```

디렉토리가 비어있거나 없으면 `prepare_calib.py`를 다시 실행하세요.

## 참고 자료

- [Sentence-BERT](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors)
- [STS Benchmark 데이터셋](https://huggingface.co/datasets/mteb/stsbenchmark-sts)
- [Mobilint 문서](https://docs.mobilint.com)
