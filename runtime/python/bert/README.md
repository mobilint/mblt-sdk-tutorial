# BERT Runtime

This tutorial explains how to run the compiled BERT sentence-similarity model with Mobilint `qbruntime`.

Before starting, complete the compilation flow in [../../../compilation/bert/README.md](../../../compilation/bert/README.md). The runtime examples in this directory expect the following files:

- `../../../compilation/bert/mxq/stsb-bert-tiny-safetensors.mxq`
- `../../../compilation/bert/weights/weight_dict.pth`

## Prerequisites

Make sure the required Python packages are available. The scripts in this directory use `torch`, `transformers`, `datasets`, `scipy`, and `tqdm`.

## Overview

This tutorial includes two kinds of runtime tasks:

1. Run example sentence-pair inference and print cosine similarities.
2. Benchmark the compiled model on the STS Benchmark test split.

For both tasks, the embedding stage runs on the host CPU and the transformer body runs on the Mobilint NPU through the local `BertMXQ` wrapper.

## Files in This Tutorial

- `inference_mxq.py`: Runs sample sentence-pair inference on the compiled MXQ model.
- `inference_original.py`: Runs the same sample inference on the original Hugging Face model for comparison.
- `benchmark_mxq.py`: Evaluates the compiled MXQ model on the STS Benchmark test split.
- `benchmark_original.py`: Evaluates the original model on the same dataset.
- `wrapper/bert_model.py`: Implements the `BertMXQ` wrapper used by the MXQ scripts.

## Run Example Inference

Run the MXQ version:

```bash
python inference_mxq.py \
    --mxq_path ../../../compilation/bert/mxq/stsb-bert-tiny-safetensors.mxq \
    --weight_path ../../../compilation/bert/weights/weight_dict.pth
```

This script tokenizes a few fixed sentence pairs, runs them through `BertMXQ`, and prints cosine similarity scores in the range `-1` to `1`.

Run the reference CPU version:

```bash
python inference_original.py
```

## Run Benchmark Evaluation

Run the MXQ benchmark:

```bash
python benchmark_mxq.py \
    --mxq_path ../../../compilation/bert/mxq/stsb-bert-tiny-safetensors.mxq \
    --weight_path ../../../compilation/bert/weights/weight_dict.pth
```

This script downloads the [STS Benchmark](https://huggingface.co/datasets/mteb/stsbenchmark-sts) test split, computes sentence similarities, and reports Pearson and Spearman correlation against the ground-truth scores.

Run the reference CPU benchmark:

```bash
python benchmark_original.py
```

## Parameters

### `inference_mxq.py`

- `--mxq_path`: Path to the compiled `.mxq` file.
- `--weight_path`: Path to the embedding weight file.

### `benchmark_mxq.py`

- `--mxq_path`: Path to the compiled `.mxq` file.
- `--weight_path`: Path to the embedding weight file.

## Expected Output

- `inference_mxq.py`: Prints cosine similarity scores for the built-in example sentence pairs.
- `benchmark_mxq.py`: Prints Pearson and Spearman correlation for the STS Benchmark test split.

Compare the MXQ results with the original-model scripts to estimate the impact of quantization and runtime execution.
