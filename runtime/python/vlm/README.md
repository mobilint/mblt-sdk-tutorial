# VLM Runtime

This tutorial explains how to run the compiled `Qwen3-VL-4B-Instruct` model on Mobilint NPU hardware.

Before starting, complete the compilation flow in [../../../compilation/vlm/README.md](../../../compilation/vlm/README.md). The runtime examples in this directory expect the following files in `../../../compilation/vlm/mxq/`:

- `Qwen3-VL-4B-Instruct_text_model.mxq`
- `Qwen3-VL-4B-Instruct_vision_transformer.mxq`
- `model.safetensors`

## Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

## Overview

This tutorial runs multimodal image-text inference through a Hugging Face-style API. The runtime flow has two stages:

1. Prepare a self-contained model folder (clone the HF repo, then swap in your compiled MXQ).
2. Run image-text generation with a prompt and an image.

The prepared folder is self-contained: it holds `config.json`, the bundled proxy classes, the tokenizer/processor, the two MXQ files, and `model.safetensors`. Inference therefore needs only `--model-folder`.

## Files in This Tutorial

- `prepare_model.py`: Creates a prepared model folder from the compilation outputs.
- `inference_mblt_model_zoo.py`: Runs image-text-to-text generation.
- `requirements.txt`: Python dependencies for this tutorial.

## Step 1: Prepare the Model Folder

```bash
python prepare_model.py \
    --repo-url https://huggingface.co/mobilint/Qwen3-VL-4B-Instruct \
    --compilation-dir ../../../compilation/vlm/mxq \
    --output-folder ./Qwen3-VL-4B-Instruct \
    --force
```

This script:

- `git clone`s the Hugging Face repo (self-contained `config.json`, proxy classes, tokenizer)
- Removes the repo's bundled `.mxq` / `.safetensors`
- Copies your compiled `.mxq` (×2) and `.safetensors` from the compilation directory
- Patches `config.json`'s `text_config.mxq_path` / `vision_config.mxq_path` to the copied filenames (the repo's core allocation is kept)

> Requires `git-lfs` so the repo's tracked files clone as real files, not pointers.
> The compiled model size must match the cloned repo (e.g. 4B artifacts ↔ 4B repo).

## Step 2: Run Inference

Run the default example:

```bash
python inference_mblt_model_zoo.py --model-folder ./Qwen3-VL-4B-Instruct
```

Useful options:

```bash
python inference_mblt_model_zoo.py --model-folder ./Qwen3-VL-4B-Instruct --image /path/to/image.jpg
python inference_mblt_model_zoo.py --model-folder ./Qwen3-VL-4B-Instruct --prompt "What objects are in this image?"
python inference_mblt_model_zoo.py --model-folder ./Qwen3-VL-4B-Instruct --max-length 1024
```

The script builds an `image-text-to-text` pipeline, feeds both the image and prompt to the model, and streams the generated output.

## NPU Core Modes

The model folder's `config.json` can be edited to change the language-model and vision-encoder core allocation.

| Mode | Description | Example language-model fields |
| --- | --- | --- |
| `single` | Run on one core | `target_cores: ["0:0"]` |
| `multi` | Multiple cores cooperate on one inference | `core_mode: "multi"`, `target_clusters: [0]` |
| `global4` | One cluster in global mode | `core_mode: "global4"`, `target_clusters: [0]` |
| `global8` | Two clusters in global mode | `core_mode: "global8"`, `target_clusters: [0, 1]` |

Use the same pattern under `vision_config` for the vision encoder.

## Parameters

### `prepare_model.py`

- `--repo-url`: Hugging Face repo to clone (self-contained config, proxy classes, tokenizer).
- `--compilation-dir`: Path to the compilation output directory (2 `.mxq` + 1 `.safetensors`).
- `--output-folder`: Destination folder (cloned repo with the compiled artifacts swapped in).
- `--force`: Remove `--output-folder` first if it already exists.

### `inference_mblt_model_zoo.py`

- `--model-folder`: Path to the self-contained model folder. The model and processor both load from here with `trust_remote_code=True`.
- `--image`: Local path or URL for the input image.
- `--prompt`: Prompt text passed along with the image.
- `--max-length`: Maximum generation length.

## Expected Output

The script streams generated text describing or answering questions about the input image.
