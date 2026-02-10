# Large Language Model Compilation

This tutorial provides detailed instructions for compiling Large Language Models (LLMs) using the Mobilint qubee compiler. The compilation process converts a standard transformer model into an optimized `.mxq` format that runs efficiently on Mobilint NPU hardware.

In this tutorial, we will use the [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) model, a 1B parameter language model developed by Meta.

## Prerequisites

Before starting, ensure you have the following installed:

- qubee SDK compiler (version >= 1.0.1 is required)
- GPU with CUDA support (recommended to reduce compilation time)
- Hugging Face account with access to Llama models (if using gated models)

Additionally, install the following packages:

```bash
pip install accelerate datasets
```

## Overview

The compilation process consists of three main steps:

1. **Model Preparation**: Download the model and extract embedding weights.
2. **Calibration Dataset Generation**: Create calibration data from the Wikitext dataset.
3. **Model Compilation**: Convert the model to `.mxq` format using the calibration data.

## Step 1: Model Preparation

Before using the model, sign up for an account on [Hugging Face](https://huggingface.co/) and accept the license agreement on the [model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).

Then, log in to Hugging Face using the following command, replacing `<your_huggingface_token>` with your actual token:

```bash
hf auth login --token <your_huggingface_token>
```

You can find your token in your [Hugging Face account settings](https://huggingface.co/settings/tokens).

Next, download the model and extract its embedding layer weights. The embedding layer is handled separately during runtime, while the rest of the model runs on the NPU.

```bash
python download_model.py \
  --repo_id meta-llama/Llama-3.2-1B-Instruct \
  --embedding ./embedding.pt
```

**What this does:**

- Downloads the specified model from the Hugging Face Hub and saves it in the cache directory.
- Extracts the input embedding layer weights.
- Saves the embedding weights to `embedding.pt`.

**Parameters:**

- `--repo_id`: Hugging Face model identifier.
- `--embedding`: Output path for the embedding weights file.

## Step 2: Calibration Dataset Preparation

Calibration data is essential for quantization during compilation. We generate this data from [Wikipedia articles](https://huggingface.co/datasets/wikimedia/wikipedia), converting text into embedding vectors that represent typical model inputs.

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

- Loads the Wikitext dataset for the specified language(s).
- Tokenizes text samples using the model's tokenizer.
- Converts tokens to embeddings using the extracted embedding layer.
- Saves calibration samples as `.npy` files.

**Parameters:**

- `--model_tag`: Model identifier (used for naming the output directory).
- `--embedding_path`: Path to the embedding weights from Step 1.
- `--tokenizer_path`: Hugging Face tokenizer identifier.
- `--output_dir`: Base directory for calibration data.
- `--min_seqlen`: Minimum sequence length (samples shorter than this are skipped).
- `--max_seqlen`: Maximum sequence length (samples are truncated to this length).
- `--max_calib`: Number of calibration samples to generate per language.

**Output Location:**
The calibration files will be saved in: `./calib/datas/meta-llama-Llama-3.2-1B-Instruct/en/`

**Multi-language Support:**
To use multiple languages, modify the `LANGUAGES` list in `generate_calib.py`:

```python
LANGUAGES = ["en", "de", "fr", "es", "it", "ja", "ko", "zh"] # Additional languages are supported
```

If using multiple languages, merge the language directories into a single directory before compilation.

## Step 3: Model Compilation

After preparing the model and calibration dataset, compile the model into the `.mxq` format. This process performs quantization and optimization for NPU execution.

```bash
python generate_mxq.py \
  --model_path meta-llama/Llama-3.2-1B-Instruct \
  --calib_data_path ./calib/datas/meta-llama-Llama-3.2-1B-Instruct/en \
  --save_path ./Llama-3.2-1B-Instruct.mxq
```

**What this does:**

- Loads the original model from Hugging Face.
- Uses calibration data to determine optimal quantization parameters.
- Compiles the model layers for NPU execution.
- Saves the compiled model in `.mxq` format.

**Parameters:**

- `--model_path`: Hugging Face model identifier.
- `--calib_data_path`: Path to the calibration data directory from Step 2.
- `--save_path`: Output path for the compiled `.mxq` model.

**Expected Output:**
The compiled model will be saved as `./Llama-3.2-1B-Instruct.mxq`.

## Next Steps

After successful compilation, proceed to `mblt-sdk-tutorial/runtime/transformers/llm/README.md` for instructions on running inference with the compiled model.

## File Summary

- `embedding.pt` - Extracted embedding layer weights.
- `Llama-3.2-1B-Instruct.mxq` - Compiled model for NPU execution.
- `calib/datas/meta-llama-Llama-3.2-1B-Instruct/en/` - Calibration dataset (128 samples).

## Extra: Model Compilation with Advanced Quantization

Starting with qubee SDK 1.0.0, we support advanced quantization techniques. This allows for flexible bit allocation and advanced parameter optimization to achieve better performance.

In the `generate_mxq.py` script, we provide pre-configured bit allocations for model layers, such as W4V8 and W4.

You can activate these modes with additional parameters:

```bash
python generate_mxq.py \
  --model_path meta-llama/Llama-3.2-1B-Instruct \
  --calib_data_path ./calib/datas/meta-llama-Llama-3.2-1B-Instruct/en \
  --save_path ./Llama-3.2-1B-Instruct.mxq \
  --bit w4
```

**Parameters:**

- `--bit`: Bit allocation for model layers. Options include `w4`, `w4v8`, and `w8`.

### Spin Quant and Rotation Transform

We highly recommend using Spin Quant with rotation transforms for better accuracy. The provided example code is designed to use this automatically when `w4` or `w4v8` quantization is selected.

After running `generate_mxq.py` with `w4` or `w4v8` quantization, a rotation matrix is generated (e.g., in `spinWeight/model/R1/global_rotation.pth`). You must rotate the embedding matrix using this rotation matrix before inference.

Run the following command to rotate the embedding matrix:

```bash
python rotate_embedding.py \
  --embedding_path ./embedding.pt \
  --rotation_matrix_path ./spinWeight/model/R1/global_rotation.pth \
  --output_path ./embedding_rotated.pt
```

**Note:** If you use `w8` quantization, rotating the embedding matrix is not required. For `w4` and `w4v8` quantization, you **must** use the rotated embedding matrix for inference.


