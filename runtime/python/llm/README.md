# LLM Runtime

This tutorial explains how to run the compiled `Llama-3.2-1B-Instruct` model on Mobilint NPU hardware.

Before starting, complete the compilation flow in [../../../compilation/llm/README.md](../../../compilation/llm/README.md). The runtime examples in this directory expect the following files:

- `../../../compilation/llm/Llama-3.2-1B-Instruct.mxq`
- `../../../compilation/llm/embedding.pt`

## Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

## Overview

This tutorial provides two runtime paths:

- `inference_mblt_model_zoo.py`: Recommended path. Uses `mblt-model-zoo` and a Hugging Face-style API.
- `inference_mxq.py`: Direct local-wrapper path. Calls `qbruntime` through `wrapper/llama_model.py`.

The recommended path needs a prepared model folder. The local-wrapper path can read the compilation outputs directly.

## Files in This Tutorial

- `prepare_model.py`: Creates a model folder for `mblt-model-zoo`.
- `inference_mblt_model_zoo.py`: Runs text generation through `mblt-model-zoo`.
- `inference_mxq.py`: Runs text generation through the local `LlamaMXQ` wrapper.
- `wrapper/llama_model.py`: Local wrapper that combines CPU-side embedding with NPU execution.
- `requirements.txt`: Python dependencies for this tutorial.

## Step 1: Prepare the Model Folder

Run this step if you want to use the recommended `mblt-model-zoo` flow.

```bash
python prepare_model.py \
    --mxq-path ../../../compilation/llm/Llama-3.2-1B-Instruct.mxq \
    --embedding-path ../../../compilation/llm/embedding.pt \
    --output-folder ./llama-mxq \
    --model-id mobilint/Llama-3.2-1B-Instruct
```

This script:

- Copies the compiled MXQ file into the output folder
- Converts `embedding.pt` into `model.safetensors`
- Downloads tokenizer and config files from Hugging Face
- Adds default NPU core allocation settings to `config.json`

## Step 2A: Run Inference with `mblt-model-zoo`

This is the recommended path.

```bash
python inference_mblt_model_zoo.py \
    --model-folder ./llama-mxq \
    --model-id mobilint/Llama-3.2-1B-Instruct
```

Useful options:

```bash
python inference_mblt_model_zoo.py --model-folder ./llama-mxq --model-id mobilint/Llama-3.2-1B-Instruct --prompt "What is quantum computing?"
python inference_mblt_model_zoo.py --model-folder ./llama-mxq --model-id mobilint/Llama-3.2-1B-Instruct --max-new-tokens 512
```

## Step 2B: Run Inference with the Local Wrapper

This path loads the compilation outputs directly and does not require `prepare_model.py`.

```bash
python inference_mxq.py \
    --mxq-path ../../../compilation/llm/Llama-3.2-1B-Instruct.mxq \
    --embedding-weight-path ../../../compilation/llm/embedding.pt
```

Useful options:

```bash
python inference_mxq.py --prompt "What is quantum computing?"
python inference_mxq.py --max-new-tokens 512
```

## NPU Core Modes

The generated `config.json` can be edited to change how the NPU cores are used.

| Mode | Description | Example config fields |
| --- | --- | --- |
| `single` | Run on one core | `target_cores: ["0:0"]` |
| `multi` | Multiple cores cooperate on one inference | `core_mode: "multi"`, `target_clusters: [0]` |
| `global4` | One cluster in global mode | `core_mode: "global4"`, `target_clusters: [0]` |
| `global8` | Two clusters in global mode | `core_mode: "global8"`, `target_clusters: [0, 1]` |

## Parameters

### `prepare_model.py`

- `--mxq-path`: Path to the compiled MXQ file.
- `--embedding-path`: Path to the embedding weight file.
- `--output-folder`: Destination folder for the prepared model.
- `--model-id`: Hugging Face model ID used for config and tokenizer download.

### `inference_mblt_model_zoo.py`

- `--model-folder`: Path to the prepared model folder.
- `--model-id`: Hugging Face model ID used for tokenizer download.
- `--prompt`: User prompt.
- `--max-new-tokens`: Maximum number of generated tokens.

### `inference_mxq.py`

- `--mxq-path`: Path to the compiled MXQ file.
- `--embedding-weight-path`: Path to the embedding weight file.
- `--prompt`: User prompt.
- `--max-new-tokens`: Maximum number of generated tokens.

## Expected Output

Both inference paths stream generated text for the prompt you provide.

In this tutorial, token embedding runs on the CPU while the transformer layers run on the Mobilint NPU.
