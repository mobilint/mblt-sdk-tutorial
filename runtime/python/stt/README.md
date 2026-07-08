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

1. Prepare a model folder from the compilation outputs.
2. Run transcription or translation on an audio file.

The prepared model folder stores the encoder MXQ, decoder MXQ, embedding weights, generation config, and NPU core allocation settings in one place.

## Files in This Tutorial

- `prepare_model.py`: Creates a Whisper model folder for runtime inference.
- `inference_mblt_model_zoo.py`: Runs transcription or translation through `mblt-model-zoo`.
- `requirements.txt`: Python dependencies for this tutorial.

## Step 1: Prepare the Model Folder

```bash
python prepare_model.py \
    --encoder-mxq ../../../compilation/stt/mxq/whisper-small_encoder.mxq \
    --decoder-mxq ../../../compilation/stt/mxq/whisper-small_decoder.mxq \
    --output-folder ./whisper-small-mxq \
    --base-model openai/whisper-small
```

This script:

- Copies the compiled encoder and decoder MXQ files
- Downloads the base Whisper configuration
- Extracts decoder embedding weights into `model.safetensors`
- Writes `config.json` with default NPU core allocation

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

The generated `config.json` can be edited to change encoder and decoder core allocation.

| Mode | Description | Example encoder fields |
| --- | --- | --- |
| `single` | Run on one core | `encoder_target_cores: ["0:0"]` |
| `multi` | Multiple cores cooperate on one inference | `encoder_core_mode: "multi"`, `encoder_target_clusters: [0]` |
| `global4` | One cluster in global mode | `encoder_core_mode: "global4"`, `encoder_target_clusters: [0]` |
| `global8` | Two clusters in global mode | `encoder_core_mode: "global8"`, `encoder_target_clusters: [0, 1]` |

Use the same pattern for the decoder with the `decoder_` prefix.

## Parameters

### `prepare_model.py`

- `--encoder-mxq`: Path to the compiled encoder MXQ file.
- `--decoder-mxq`: Path to the compiled decoder MXQ file.
- `--output-folder`: Destination folder for the prepared model.
- `--base-model`: Hugging Face base model ID used for config and embedding extraction.
- `--model-id`: Hugging Face model ID stored in the prepared config.

### `inference_mblt_model_zoo.py`

- `--audio`: Path to the input audio file.
- `--model-folder`: Path to the prepared model folder.
- `--model-id`: Hugging Face model ID used for processor download.
- `--language`: Optional source language code such as `en`, `ko`, or `ja`.
- `--task`: `transcribe` or `translate`.

## Expected Output

The script prints the final transcription or translation result after generation completes.

Whisper supports many languages, and the example script can either auto-detect the language or accept it explicitly through `--language`.
