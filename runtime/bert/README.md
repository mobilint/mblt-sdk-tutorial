# Bidirectional Encoder Representations from Transformers (BERT)

This tutorial provides detailed instructions for performing inference with the compiled BERT model.

This guide follows `mblt-sdk-tutorial/compilation/bert/README.md`. We assume you have successfully compiled the model and have the following files ready:

- `compilation/bert/mxq/stsb-bert-tiny-safetensors.mxq` - Compiled model file
- `compilation/bert/weights/weight_dict.pth` - Embedding layer weights

## Running Inference

### MXQ Model (NPU)

Run inference on the compiled MXQ model using Mobilint NPU. This script computes cosine similarity between dummy sentence pairs using the `BertMXQ` wrapper class (see `wrapper/bert_model.py`).

```bash
python inference_mxq.py \
    --mxq_path ../../compilation/bert/mxq/stsb-bert-tiny-safetensors.mxq \
    --weight_path ../../compilation/bert/weights/weight_dict.pth
```

The wrapper class (`wrapper/bert_model.py`) processes input embeddings (word, token type, position) on the CPU and executes the remaining transformer layers on the Mobilint NPU via `qbruntime`.

### Original Model (CPU)

Run the same inference using the original HuggingFace model for comparison:

```bash
python inference_original.py
```

## Performance Evaluation

To accurately evaluate model quality, we compute sentence similarity for all pairs in the [STS Benchmark](https://huggingface.co/datasets/mteb/stsbenchmark-sts) test set, then measure Pearson and Spearman correlation between ground-truth scores and model predictions.

### MXQ Model (NPU)

```bash
python benchmark_mxq.py \
    --mxq_path ../../compilation/bert/mxq/stsb-bert-tiny-safetensors.mxq \
    --weight_path ../../compilation/bert/weights/weight_dict.pth
```

### Original Model (CPU)

```bash
python benchmark_original.py
```

Compare the correlation coefficients from both scripts to assess the quantization impact on model accuracy.

## Command Line Arguments

### `inference_mxq.py`

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--mxq_path` | `../../compilation/bert/mxq/stsb-bert-tiny-safetensors.mxq` | Path to the compiled MXQ file |
| `--weight_path` | `../../compilation/bert/weights/weight_dict.pth` | Path to the embedding weight file |

### `benchmark_mxq.py`

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--mxq_path` | `../../compilation/bert/mxq/stsb-bert-tiny-safetensors.mxq` | Path to the compiled MXQ file |
| `--weight_path` | `../../compilation/bert/weights/weight_dict.pth` | Path to the embedding weight file |

## File Structure

```text
bert/
├── inference_mxq.py          # MXQ inference (NPU)
├── inference_original.py     # Original model inference (CPU)
├── benchmark_mxq.py          # STS Benchmark evaluation (NPU)
├── benchmark_original.py     # STS Benchmark evaluation (CPU)
└── wrapper/
    └── bert_model.py         # BertMXQ wrapper for qbruntime
```

## References

- [Compilation Tutorial](../../compilation/bert/README.md)
- [stsb-bert-tiny-safetensors Model Card](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors)
- [STS Benchmark Dataset](https://huggingface.co/datasets/mteb/stsbenchmark-sts)
- [Mobilint Documentation](https://docs.mobilint.com)
