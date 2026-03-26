# Bidirectional Encoder Representations from Transformers (BERT)

This tutorial provides detailed instructions for performing inference with the compiled BERT model.

This guide follows `mblt-sdk-tutorial/compilation/bert/README.md`. We assume you have successfully compiled the model and have the following files ready:

- `./stsb-bert-tiny-safetensors.mxq` - Compiled model file
- `./weight_dict.pth` - Embedding layer weights

## Running Inference

### MXQ Model (NPU)

Run inference on the compiled MXQ model using Mobilint NPU. This script computes cosine similarity between dummy sentence pairs using the `BertMXQModel` wrapper class (see `wrapper/bertmxq.py`).

```bash
python inference_mxq.py \
    --mxq_path ../../compilation/bert/mxq/stsb-bert-tiny-safetensors.mxq \
    --weight_path ../../compilation/bert/weights/weight_dict.pth
```

The wrapper class (`wrapper/bertmxq.py`) processes input embeddings (word, token type, position) on the CPU and executes the remaining transformer layers on the Mobilint NPU via `qbruntime`.

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
