# VLM Runtime

This tutorial explains how to run the compiled `Qwen2-VL-2B-Instruct` model on Mobilint NPU hardware.

Before starting, complete the compilation flow in [../../../compilation/vlm/README.md](../../../compilation/vlm/README.md). The runtime examples in this directory expect the following files in `../../../compilation/vlm/mxq/`:

- `Qwen2-VL-2B-Instruct_text_model.mxq`
- `Qwen2-VL-2B-Instruct_vision_transformer.mxq`
- `config.json`
- `model.safetensors`

## Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

## Overview

This tutorial uses `mblt-model-zoo` to run multimodal image-text inference through a Hugging Face-style API. The runtime flow has two stages:

1. Prepare a model folder from the compilation outputs.
2. Run image-text generation with a prompt and an image.

The prepared model folder keeps the text model MXQ, vision encoder MXQ, configuration, safetensors weights, and NPU core allocation in one place.

## Files in This Tutorial

- `prepare_model.py`: Creates a prepared model folder from the compilation outputs.
- `inference_mblt_model_zoo.py`: Runs image-text-to-text generation.
- `requirements.txt`: Python dependencies for this tutorial.

## Step 1: Prepare the Model Folder

```bash
python prepare_model.py \
    --compilation-dir ../../../compilation/vlm/mxq \
    --output-folder ./qwen2-vl-mxq \
    --model-id mobilint/Qwen2-VL-2B-Instruct
```

This script:

- Copies the compiled MXQ files, `config.json`, and `model.safetensors`
- Adds default NPU core allocation to `config.json`
- Updates `_name_or_path` so the prepared folder matches the intended model ID

## Step 2: Run Inference

Run the default example:

```bash
python inference_mblt_model_zoo.py \
    --model-folder ./qwen2-vl-mxq \
    --model-id mobilint/Qwen2-VL-2B-Instruct
```

Useful options:

```bash
python inference_mblt_model_zoo.py --model-folder ./qwen2-vl-mxq --model-id mobilint/Qwen2-VL-2B-Instruct --image /path/to/image.jpg
python inference_mblt_model_zoo.py --model-folder ./qwen2-vl-mxq --model-id mobilint/Qwen2-VL-2B-Instruct --prompt "What objects are in this image?"
python inference_mblt_model_zoo.py --model-folder ./qwen2-vl-mxq --model-id mobilint/Qwen2-VL-2B-Instruct --max-length 1024
```

The script builds an `image-text-to-text` pipeline, feeds both the image and prompt to the model, and streams the generated output.

## NPU Core Modes

The generated `config.json` can be edited to change the language-model and vision-encoder core allocation.

| Mode | Description | Example language-model fields |
| --- | --- | --- |
| `single` | Run on one core | `target_cores: ["0:0"]` |
| `multi` | Multiple cores cooperate on one inference | `core_mode: "multi"`, `target_clusters: [0]` |
| `global4` | One cluster in global mode | `core_mode: "global4"`, `target_clusters: [0]` |
| `global8` | Two clusters in global mode | `core_mode: "global8"`, `target_clusters: [0, 1]` |

Use the same pattern under `vision_config` for the vision encoder.

## Parameters

### `prepare_model.py`

- `--compilation-dir`: Path to the compilation output directory.
- `--output-folder`: Destination folder for the prepared model.
- `--model-id`: Hugging Face model ID stored in the prepared config.

### `inference_mblt_model_zoo.py`

- `--model-folder`: Path to the prepared model folder.
- `--model-id`: Hugging Face model ID used for processor download.
- `--image`: Local path or URL for the input image.
- `--prompt`: Prompt text passed along with the image.
- `--max-length`: Maximum generation length.

## Expected Output

The script streams generated text describing or answering questions about the input image.
