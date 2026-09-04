# BERT Python Runtime

Complete the [BERT compilation tutorial](../../../compilation/bert/README.md) first.

The runtime uses the single model directory created by `compilation/bert/prepare_model.py`.

- `../../../compilation/bert/bert-mxq`

Run all commands from this directory.

## Prerequisites

```bash
pip install -r requirements.txt
```

## MXQ Inference

```bash
python inference_mxq.py \
  --model-folder ../../../compilation/bert/bert-mxq
```

The CPU computes input embeddings from the downloaded original `model.safetensors`.
The NPU runs the BERT encoder, and the CPU applies Sentence-BERT mean pooling.
The current MXQ runtime supports one unpadded sentence at a time.

## Original Model Inference

```bash
python inference_original.py \
  --model-folder ../../../compilation/bert/bert-mxq
```

## MXQ Benchmark

```bash
python benchmark_mxq.py \
  --model-folder ../../../compilation/bert/bert-mxq
```

## Original Model Benchmark

```bash
python benchmark_original.py \
  --model-folder ../../../compilation/bert/bert-mxq
```

Both benchmark scripts report Pearson and Spearman correlations on the STS Benchmark test split.
