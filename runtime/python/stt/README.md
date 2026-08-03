# STT Runtime

This tutorial explains how to run the compiled Whisper speech-to-text model on Mobilint NPU hardware.

Before starting, complete the compilation flow in [../../../compilation/stt/README.md](../../../compilation/stt/README.md). The runtime examples in this directory expect the following files:

- `../../../compilation/stt/mxq/whisper-small_encoder.mxq`
- `../../../compilation/stt/mxq/whisper-small_decoder.mxq`
- `../../../compilation/stt/audio_files/`

## Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

## Overview

This tutorial uses `mblt-model-zoo` to run Whisper through a Hugging Face-style API. The runtime flow has two stages:

1. Prepare a self-contained model folder (download the HF repo, then swap in your compiled MXQ).
2. Run transcription or translation on an audio file.

The prepared folder is self-contained: it holds `config.json`, the bundled proxy classes, the tokenizer/processor, generation config, the encoder/decoder MXQ files, and `model.safetensors` (decoder embedding weights).

## Files in This Tutorial

- `prepare_model.py`: Creates a Whisper model folder for runtime inference.
- `inference_mblt_model_zoo.py`: Runs transcription or translation through `mblt-model-zoo`.
- `requirements.txt`: Python dependencies for this tutorial.

## Step 1: Prepare the Model Folder

```bash
python prepare_model.py \
    --repo-id mobilint/whisper-small \
    --compilation-dir ../../../compilation/stt/mxq \
    --output-folder ./whisper-small-mxq \
    --force
```

This script:

- Downloads the Hugging Face repo via `huggingface_hub.snapshot_download` (self-contained `config.json`, proxy classes, tokenizer, generation config, and `model.safetensors`), skipping only the repo's own `.mxq`
- Copies your compiled encoder and decoder `.mxq` from the compilation directory
- Patches `config.json`'s `encoder_mxq_path` / `decoder_mxq_path` to the copied filenames (the repo's core allocation is kept)

> No `git-lfs` needed — `snapshot_download` fetches real files (`huggingface_hub` is installed with `mblt-model-zoo[transformers]`).
> Unlike the VLM flow, `model.safetensors` is kept from the repo: for Whisper it holds the decoder embedding weights (run on CPU), not a compilation output.

## Step 2: Run Inference

Run the default transcription example:

```bash
python inference_mblt_model_zoo.py \
    --audio ../../../compilation/stt/audio_files/en_us_0000.wav \
    --model-folder ./whisper-small-mxq \
    --model-id mobilint/whisper-small
```

Useful options:

```bash
python inference_mblt_model_zoo.py --audio audio.wav --model-folder ./whisper-small-mxq --model-id mobilint/whisper-small --language en
python inference_mblt_model_zoo.py --audio audio.wav --model-folder ./whisper-small-mxq --model-id mobilint/whisper-small --task translate
```

The script loads audio with `librosa`, resamples it to `16 kHz`, runs generation through `AutoModelForSpeechSeq2Seq`, and prints the decoded text.

## NPU Core Modes

The model folder's `config.json` (from the downloaded repo, which defaults to `global8`) can be edited to change encoder and decoder core allocation.

| Mode | Description | Example encoder fields |
| --- | --- | --- |
| `single` | Run on one core | `encoder_target_cores: ["0:0"]` |
| `multi` | Multiple cores cooperate on one inference | `encoder_core_mode: "multi"`, `encoder_target_clusters: [0]` |
| `global4` | One cluster in global mode | `encoder_core_mode: "global4"`, `encoder_target_clusters: [0]` |
| `global8` | Two clusters in global mode | `encoder_core_mode: "global8"`, `encoder_target_clusters: [0, 1]` |

Use the same pattern for the decoder with the `decoder_` prefix.

## Parameters

### `prepare_model.py`

- `--repo-id`: Hugging Face repo id to download (self-contained config, proxy classes, tokenizer, embeddings).
- `--compilation-dir`: Path to the compilation output directory holding the 2 `.mxq` (encoder and decoder).
- `--output-folder`: Destination folder (downloaded repo with the compiled MXQ swapped in).
- `--force`: Remove `--output-folder` first if it already exists.

### `inference_mblt_model_zoo.py`

- `--audio`: Path to the input audio file.
- `--model-folder`: Path to the prepared model folder.
- `--model-id`: Hugging Face model ID used for processor download.
- `--language`: Optional source language code such as `en`, `ko`, or `ja`.
- `--task`: `transcribe` or `translate`.

## Expected Output

The script prints the final transcription or translation result after generation completes.

Whisper supports many languages, and the example script can either auto-detect the language or accept it explicitly through `--language`.
