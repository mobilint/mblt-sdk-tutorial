# BERT Python 런타임

먼저 [BERT 컴파일 튜토리얼](../../../compilation/bert/README.KR.md)을 완료합니다.

런타임은 `compilation/bert/prepare_model.py`가 만든 하나의 모델 디렉터리를 사용합니다.

- `../../../compilation/bert/bert-mxq`

모든 명령은 이 디렉토리에서 실행합니다.

## 사전 준비

```bash
pip install -r requirements.txt
```

## MXQ 추론

```bash
python inference_mxq.py \
  --model-folder ../../../compilation/bert/bert-mxq
```

입력 임베딩은 다운로드한 원본 `model.safetensors`를 사용해 CPU에서 계산합니다.
BERT encoder는 NPU에서 실행하고 Sentence-BERT mean pooling은 CPU에서 계산합니다.
현재 MXQ 런타임은 패딩이 없는 단일 문장 입력만 지원합니다.

## 원본 모델 추론

```bash
python inference_original.py \
  --model-folder ../../../compilation/bert/bert-mxq
```

## MXQ 벤치마크

```bash
python benchmark_mxq.py \
  --model-folder ../../../compilation/bert/bert-mxq
```

## 원본 모델 벤치마크

```bash
python benchmark_original.py \
  --model-folder ../../../compilation/bert/bert-mxq
```

두 벤치마크는 STS Benchmark 테스트 세트에 대한 Pearson 및 Spearman 상관계수를 출력합니다.
