# Speech-to-Text Model Inference (Whisper)

This tutorial provides step-by-step instructions for running inference with compiled Whisper speech-to-text models using the Mobilint qbruntime.

This guide is a continuation of [mblt-sdk-tutorial/compilation/transformers/stt/README.md](file:///workspace/mblt-sdk-tutorial/compilation/transformers/stt/README.md). It is assumed that you have successfully compiled the model.

## Prerequisites

Before running inference, ensure you have the following components installed and available:

- Compiled Whisper MXQ models (encoder and decoder)
- `qbruntime` library (to access the NPU accelerator)
- Python packages: `transformers`, `torch`, `librosa`, `safetensors`

## Files

| File | Description |
|------|-------------|
| `mblt-whisper.py` | Hugging Face-compatible Mobilint Whisper model implementation |
| `prepare_model.py` | Script to prepare the model directory with necessary configuration files |
| `inference_mxq.py` | Main inference script for audio transcription and translation |

## Quick Start

### Step 1: Prepare Model Folder

First, execute the `prepare_model.py` script to organize the compiled MXQ files and generate the required configuration:

```bash
python prepare_model.py \
    --encoder-mxq ../../../compilation/transformers/stt/compilation/compiled/whisper-small_encoder.mxq \
    --decoder-mxq ../../../compilation/transformers/stt/compilation/compiled/whisper-small_decoder.mxq \
    --output-folder ./whisper-small-mxq \
    --base-model openai/whisper-small
```

This script performs the following actions:
- Copies the compiled MXQ files to the specified output folder.
- Downloads the processor (tokenizer and feature extractor) from Hugging Face.
- Extracts and saves the embedding weights from the base model.
- Generates the configuration files required for the Mobilint Whisper model.

### Step 2: Run Inference

Execute speech-to-text inference on an audio file using `inference_mxq.py`:

```bash
python inference_mxq.py \
    --audio /path/to/audio.wav \
    --model-folder ./whisper-small-mxq
```

## Usage Options

### Basic Transcription

To transcribe an audio file:

```bash
python inference_mxq.py --audio audio.wav --model-folder ./whisper-small-mxq
```

### Specify Language

To specify the source language (e.g., English):

```bash
python inference_mxq.py --audio audio.wav --model-folder ./whisper-small-mxq --language en
```

### Translation to English

To translate the spoken audio to English:

```bash
python inference_mxq.py --audio audio.wav --model-folder ./whisper-small-mxq --task translate
```

### Use Pipeline API

To use the Hugging Face pipeline API (recommended for longer audio files):

```bash
python inference_mxq.py --audio audio.wav --model-folder ./whisper-small-mxq --use-pipeline
```

## Command Line Arguments

### `prepare_model.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--encoder-mxq` | `../../../compilation/transformers/stt/compilation/compiled/whisper-small_encoder.mxq` | Path to the compiled encoder MXQ file |
| `--decoder-mxq` | `../../../compilation/transformers/stt/compilation/compiled/whisper-small_decoder.mxq` | Path to the compiled decoder MXQ file |
| `--output-folder` | `./whisper-small-mxq` | Destination folder for the prepared model |
| `--base-model` | `openai/whisper-small` | Hugging Face model ID used for base configuration |

### `inference_mxq.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--audio` | `../../../compilation/transformers/stt/data/audio_files/en_us_0000.wav` | Path to the input audio file |
| `--model-folder` | `./whisper-small-mxq` | Path to the prepared model folder |
| `--language` | `None` (auto-detect) | Source language code (e.g., `en`, `ko`, `ja`) |
| `--task` | `transcribe` | Task to perform: `transcribe` or `translate` |
| `--use-pipeline` | `False` | If set, uses the Hugging Face pipeline API instead of manual inference |

## Supported Languages

Whisper supports over 99 languages. Common language codes include:
- `en` - English
- `ko` - Korean
- `ja` - Japanese
- `zh` - Chinese
- `es` - Spanish
- `fr` - French
- `de` - German

## Architecture

The implementation leverages Hugging Face's `AutoModel` classes for seamless integration:

```
┌─────────────────────────────────────────────────────────┐
│                   inference_mxq.py                      │
│  (Loads model via AutoModelForSpeechSeq2Seq)            │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   mblt-whisper.py                       │
│  ┌─────────────────────────────────────────────────┐    │
│  │  MobilintWhisperForConditionalGeneration        │    │
│  │  ├── MobilintWhisperEncoder (encoder.mxq)       │    │
│  │  └── MobilintWhisperDecoder (decoder.mxq)       │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  Mobilint Accelerator                   │
│                      (qbruntime)                        │
└─────────────────────────────────────────────────────────┘
```

## Notes

- Audio files are automatically resampled to 16kHz.
- The model processes audio in chunks of up to 30 seconds.
- For audio files longer than 30 seconds, it is recommended to use the pipeline API (`--use-pipeline`), which automatically handles chunking.
