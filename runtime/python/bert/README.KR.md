# BERT 런타임

이 튜토리얼은 Mobilint `qbruntime`를 사용해 컴파일된 BERT 문장 유사도 모델을 실행하는 방법을 설명합니다.

시작하기 전에 [../../../compilation/bert/README.KR.md](../../../compilation/bert/README.KR.md)의 컴파일 과정을 먼저 완료하세요. 이 디렉토리의 런타임 예제는 다음 파일이 준비되어 있다고 가정합니다.

- `../../../compilation/bert/mxq/stsb-bert-tiny-safetensors.mxq`
- `../../../compilation/bert/weights/weight_dict.pth`

## 사전 준비

이 디렉토리의 스크립트는 `torch`, `transformers`, `datasets`, `scipy`, `tqdm` 등의 Python 패키지를 사용합니다. 필요한 패키지가 설치되어 있는지 확인하세요.

## 개요

이 튜토리얼은 두 가지 런타임 작업을 제공합니다.

1. 예제 문장 쌍에 대해 추론을 실행하고 cosine similarity를 출력합니다.
2. 컴파일된 모델을 STS Benchmark 테스트셋으로 평가합니다.

두 작업 모두 입력 임베딩 단계는 호스트 CPU에서 실행되고, 트랜스포머 본체는 로컬 `BertMXQ` 래퍼를 통해 Mobilint NPU에서 실행됩니다.

## 이 튜토리얼의 파일

- `inference_mxq.py`: 컴파일된 MXQ 모델로 예제 문장 쌍 추론을 실행합니다.
- `inference_original.py`: 비교를 위해 원본 Hugging Face 모델로 동일한 추론을 실행합니다.
- `benchmark_mxq.py`: 컴파일된 MXQ 모델을 STS Benchmark 테스트셋으로 평가합니다.
- `benchmark_original.py`: 원본 모델을 동일한 데이터셋으로 평가합니다.
- `wrapper/bert_model.py`: MXQ 스크립트에서 사용하는 `BertMXQ` 래퍼를 구현합니다.

## 예제 추론 실행

MXQ 버전 실행:

```bash
python inference_mxq.py \
    --mxq_path ../../../compilation/bert/mxq/stsb-bert-tiny-safetensors.mxq \
    --weight_path ../../../compilation/bert/weights/weight_dict.pth
```

이 스크립트는 고정된 몇 개의 문장 쌍을 토크나이즈한 뒤 `BertMXQ`에 전달하고, `-1`에서 `1` 사이의 cosine similarity 점수를 출력합니다.

비교용 CPU 버전 실행:

```bash
python inference_original.py
```

## 벤치마크 평가 실행

MXQ 벤치마크 실행:

```bash
python benchmark_mxq.py \
    --mxq_path ../../../compilation/bert/mxq/stsb-bert-tiny-safetensors.mxq \
    --weight_path ../../../compilation/bert/weights/weight_dict.pth
```

이 스크립트는 [STS Benchmark](https://huggingface.co/datasets/mteb/stsbenchmark-sts) 테스트 split을 다운로드한 뒤 문장 유사도를 계산하고, 정답 점수와의 Pearson 및 Spearman 상관 계수를 출력합니다.

비교용 CPU 벤치마크 실행:

```bash
python benchmark_original.py
```

## 파라미터

### `inference_mxq.py`

- `--mxq_path`: 컴파일된 `.mxq` 파일 경로
- `--weight_path`: 임베딩 가중치 파일 경로

### `benchmark_mxq.py`

- `--mxq_path`: 컴파일된 `.mxq` 파일 경로
- `--weight_path`: 임베딩 가중치 파일 경로

## 예상 출력

- `inference_mxq.py`: 내장된 예제 문장 쌍에 대한 cosine similarity 점수를 출력합니다.
- `benchmark_mxq.py`: STS Benchmark 테스트 split에 대한 Pearson 및 Spearman 상관 계수를 출력합니다.

양자화 및 런타임 실행의 영향을 확인하려면 MXQ 결과를 원본 모델 스크립트와 비교하세요.
