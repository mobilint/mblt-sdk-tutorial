# Bidirectional Encoder Representations from Transformers (BERT)

이 튜토리얼은 컴파일된 BERT 모델을 사용하여 추론(Inference)을 수행하는 세부 지침을 제공합니다.

이 가이드는 `mblt-sdk-tutorial/compilation/bert/README.md`의 내용을 이어받아 진행됩니다. 모델이 성공적으로 컴파일되었으며 다음 파일들이 준비되어 있다고 가정합니다.

- `compilation/bert/mxq/stsb-bert-tiny-safetensors.mxq` - 컴파일된 모델 파일
- `compilation/bert/weights/weight_dict.pth` - 임베딩 레이어 가중치

## 추론 실행

### MXQ 모델 (NPU)

컴파일된 MXQ 모델을 사용하여 Mobilint NPU에서 추론을 실행합니다. 이 스크립트는 `BertMXQ` 래퍼 클래스(`wrapper/bert_model.py` 참조)를 사용하여 더미 문장 쌍 간의 코사인 유사도를 계산합니다.

```bash
python inference_mxq.py \
    --mxq_path ../../compilation/bert/mxq/stsb-bert-tiny-safetensors.mxq \
    --weight_path ../../compilation/bert/weights/weight_dict.pth
```

래퍼 클래스(`wrapper/bert_model.py`)는 입력 임베딩(word, token type, position)을 CPU에서 처리하고, 나머지 트랜스포머 레이어는 `qbruntime`을 통해 Mobilint NPU에서 실행합니다.

### 원본 모델 (CPU)

비교를 위해 원본 HuggingFace 모델로 동일한 추론을 실행합니다:

```bash
python inference_original.py
```

## 성능 평가

모델 품질을 정확히 평가하기 위해 [STS Benchmark](https://huggingface.co/datasets/mteb/stsbenchmark-sts) 테스트셋의 모든 문장 쌍에 대한 유사도를 계산하고, 정답 점수와 모델 예측 간의 피어슨(Pearson) 및 스피어먼(Spearman) 상관 계수를 측정합니다.

### MXQ 모델 (NPU)

```bash
python benchmark_mxq.py \
    --mxq_path ../../compilation/bert/mxq/stsb-bert-tiny-safetensors.mxq \
    --weight_path ../../compilation/bert/weights/weight_dict.pth
```

### 원본 모델 (CPU)

```bash
python benchmark_original.py
```

두 스크립트의 상관 계수를 비교하여 양자화가 모델 정확도에 미치는 영향을 평가할 수 있습니다.

## 커맨드 라인 인자

### `inference_mxq.py`

| 인자 | 기본값 | 설명 |
| --- | ----- | --- |
| `--mxq_path` | `../../compilation/bert/mxq/stsb-bert-tiny-safetensors.mxq` | 컴파일된 MXQ 파일 경로 |
| `--weight_path` | `../../compilation/bert/weights/weight_dict.pth` | 임베딩 가중치 파일 경로 |

### `benchmark_mxq.py`

| 인자 | 기본값 | 설명 |
| --- | ----- | --- |
| `--mxq_path` | `../../compilation/bert/mxq/stsb-bert-tiny-safetensors.mxq` | 컴파일된 MXQ 파일 경로 |
| `--weight_path` | `../../compilation/bert/weights/weight_dict.pth` | 임베딩 가중치 파일 경로 |

## 파일 구조

```text
bert/
├── inference_mxq.py          # MXQ 추론 (NPU)
├── inference_original.py     # 원본 모델 추론 (CPU)
├── benchmark_mxq.py          # STS Benchmark 평가 (NPU)
├── benchmark_original.py     # STS Benchmark 평가 (CPU)
└── wrapper/
    └── bert_model.py         # qbruntime용 BertMXQ 래퍼
```

## 참조

- [컴파일 튜토리얼](../../compilation/bert/README.md)
- [stsb-bert-tiny-safetensors 모델 카드](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors)
- [STS Benchmark 데이터셋](https://huggingface.co/datasets/mteb/stsbenchmark-sts)
- [Mobilint Documentation](https://docs.mobilint.com)
