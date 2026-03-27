# Large Language Model (LLM) Compilation

This tutorial provides instructions for compiling Large Language Models (LLMs) using the Mobilint qbcompiler.

In this tutorial, we will use the [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) model, a 1B parameter language model developed by Meta.

## Overview

The compilation process consists of three main steps:

1. **Model Preparation**: Download the model and extract embedding weights
2. **Calibration Data Generation**: Create calibration data from Wikipedia articles
3. **Model Compilation**: Compile the model to `.mxq` format with 8-bit quantization

## Prerequisites

- qbcompiler SDK (version >= 1.0.1)
- (optional) CUDA-capable GPU for faster compilation
- Hugging Face account with access to Llama models

```bash
pip install -r requirements.txt
```

## Step 1: Download Model

Sign up on [Hugging Face](https://huggingface.co/) and accept the license on the [model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), then log in:

```bash
huggingface-cli login --token <your_huggingface_token>
```

Download the model and extract its embedding weights. The embedding layer runs on CPU at inference time, while the rest of the model runs on NPU.

```bash
python download_model.py \
  --repo-id meta-llama/Llama-3.2-1B-Instruct \
  --embedding-path ./embedding.pt
```

**Output:**

- `embedding.pt` — Embedding weight matrix `[vocab_size, embed_dim]`

## Step 2: Generate Calibration Data

Generate calibration data from [Wikipedia articles](https://huggingface.co/datasets/wikimedia/wikipedia). Text is tokenized and converted to embedding vectors for quantization calibration.

```bash
python generate_calib.py \
  --model-tag meta-llama/Llama-3.2-1B-Instruct \
  --embedding-path ./embedding.pt \
  --tokenizer-path meta-llama/Llama-3.2-1B-Instruct \
  --output-dir ./calibration_data
```

**Output:**

- `./calibration_data/datas/meta-llama-Llama-3.2-1B-Instruct/en/` — 128 calibration samples (`.npy`)

## Step 3: Compile Model (8-bit)

Compile the model to `.mxq` format with 8-bit quantization.

```bash
python generate_mxq.py \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --calib-data-path ./calibration_data/datas/meta-llama-Llama-3.2-1B-Instruct/en \
  --save-path ./Llama-3.2-1B-Instruct.mxq
```

**Output:**

- `Llama-3.2-1B-Instruct.mxq` — Compiled model for NPU execution

## Next Steps

After compilation, see `mblt-sdk-tutorial/runtime/transformers/llm/README.md` for inference instructions.

---

## Advanced: 4-Bit Quantization

4-bit quantization reduces model size further but requires SpinQuant rotation and weight scale search to maintain accuracy. This generates `spinWeight/` rotation matrices that must be applied to the embedding layer before inference.

### Step 1: Compile with 4-bit

Use `generate_mxq_4bit.py` instead of `generate_mxq.py`:

```bash
python generate_mxq_4bit.py \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --calib-data-path ./calibration_data/datas/meta-llama-Llama-3.2-1B-Instruct/en \
  --save-path ./Llama-3.2-1B-Instruct_w4.mxq \
  --bit w4
```

- `--bit`: Bit allocation preset. `w4` (all 4-bit, default) or `w4v8` (4-bit except value kept at 8-bit for accuracy).

**Output:**

- `Llama-3.2-1B-Instruct_w4.mxq` — 4-bit compiled model
- `spinWeight/model/R1/global_rotation.pth` — SpinQuant rotation matrix

### Step 2: Rotate Embedding

The SpinQuant rotation transforms the model's internal weight space. Since the embedding layer runs on CPU (not compiled into MXQ), it must be pre-rotated with the same rotation matrix.

```bash
python get_rotation_emb.py \
  --embedding-path ./embedding.pt \
  --rotation-matrix-path ./spinWeight/model/R1/global_rotation.pth \
  --output-path ./embedding_rot.pt
```

**Output:**

- `embedding_rot.pt` — Rotated embedding weights for 4-bit inference

> **Note:** Use `embedding_rot.pt` (not `embedding.pt`) at inference time with 4-bit models. 8-bit models do not require embedding rotation.
