# Large Language Model Compilation

This tutorial provides detailed instructions for compiling large language models using the Mobilint QB compiler. The compilation process converts a standard transformer model into an optimized `.mxq` format that can run efficiently on Mobilint NPU hardware.

In this tutorial, we will use the [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) model, a 1B parameter language model developed by Meta.

## Prerequisites

Before starting, ensure you have the following installed:

- qubee SDK compiler installed (version >= 0.11 required)
- GPU with CUDA support (recommended for reducing compilation time)
- HuggingFace account with access to Llama models (if using gated models)

## Overview

The compilation process consists of three main steps:

1. **Model Preparation**: Download the model and extract embedding weights
2. **Calibration Dataset Generation**: Create calibration data from Wikitext dataset
3. **Model Compilation**: Convert the model to `.mxq` format using calibration data

## Step 1: Model Preparation

Before using the model, sign up for an account on [HuggingFace](https://huggingface.co/) and sign the agreement to use the model on the [model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).

Then, login to HuggingFace using the following command and replace <your_huggingface_token> with your actual HuggingFace token:

```bash
hf auth login --token <your_huggingface_token>
```

If you are not sure about your HuggingFace token, you can find it in your [HuggingFace account settings](https://huggingface.co/settings/tokens).

Then, download the model from HuggingFace and extract its embedding layer weights. The embedding layer is used separately during runtime while the rest of the model runs on the NPU.

```bash
python download_model.py \
  --repo_id meta-llama/Llama-3.2-1B-Instruct \
  --embedding ./embedding.pt
```

**What this does:**

- Downloads the specified model from HuggingFace Hub
- Extracts the input embedding layer weights
- Saves the embedding weight to `embedding.pt`

**Parameters:**

- `--repo_id`: HuggingFace model identifier
- `--embedding`: Output path for embedding weights file

## Step 2: Calibration Dataset Preparation

Calibration data is essential for quantization during compilation. We generate this data from Wikipedia articles, converting text into embedding vectors that represent typical model inputs.

```bash
python generate_calib.py \
  --model_tag meta-llama/Llama-3.2-1B-Instruct \
  --embedding_path ./embedding.pt \
  --tokenizer_path meta-llama/Llama-3.2-1B-Instruct \
  --output_dir ./calib \
  --min_seqlen 512 \
  --max_seqlen 2048 \
  --max_calib 128
```

**What this does:**

- Loads Wikitext dataset for specified language(s)
- Tokenizes text samples using the model's tokenizer
- Converts tokens to embeddings using the extracted embedding layer
- Saves calibration samples as `.npy` files

**Parameters:**

- `--model_tag`: Model identifier (used for output directory naming)
- `--embedding_path`: Path to the embedding weights from Step 1
- `--tokenizer_path`: HuggingFace tokenizer identifier
- `--output_dir`: Base directory for calibration data
- `--min_seqlen`: Minimum sequence length (samples shorter than this are skipped)
- `--max_seqlen`: Maximum sequence length (samples are truncated to this length)
- `--max_calib`: Number of calibration samples to generate per language

**Output Location:**
The calibration files will be saved in: `./calib/datas/meta-llama-Llama-3.2-1B-Instruct/en/`

**Multi-language Support:**
To use multiple languages, modify the `LANGUAGES` list in `generate_calib.py`:

```python
LANGUAGES = ["en", "de", "fr", "es", "it", "ja", "ko", "zh"]
```

If using multiple languages, merge the language directories into a single directory before compilation.

## Step 3: Model Compilation

After preparing the model and calibration dataset, compile the model to `.mxq` format. This process performs quantization and optimization for NPU execution.

```bash
python generate_mxq.py \
  --model_path meta-llama/Llama-3.2-1B-Instruct \
  --calib_data_path ./calib/datas/meta-llama-Llama-3.2-1B-Instruct/en \
  --save_path ./Llama-3.2-1B-Instruct.mxq
```

**What this does:**

- Loads the original model from HuggingFace
- Uses calibration data to determine optimal quantization parameters
- Compiles the model layers for NPU execution
- Saves the compiled model as `.mxq` format

**Parameters:**

- `--model_path`: HuggingFace model identifier
- `--calib_data_path`: Path to calibration data directory from Step 2
- `--save_path`: Output path for compiled `.mxq` model

**Expected Output:**
The compiled model will be saved as `./Llama-3.2-1B-Instruct.mxq`

## Next Steps

After successful compilation, proceed to `mblt-sdk-tutorial/runtime/transformers/llm/README.md` for instructions on running inference with the compiled model.

## File Summary

- `embedding.pt` - Extracted embedding layer weights
- `Llama-3.2-1B-Instruct.mxq` - Compiled model for NPU execution
- `calib/datas/meta-llama-Llama-3.2-1B-Instruct/en/` - Calibration dataset (128 samples)
