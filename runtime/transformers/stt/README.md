# Whisper Speech-to-Text Runtime

This tutorial demonstrates how to run inference on compiled Whisper MXQ models using the Mobilint accelerator.

## Prerequisites

- Compiled Whisper MXQ models (encoder and decoder) from the [compilation tutorial](../../../compilation/transformers/stt/README.md)
- Mobilint SDK with `maccel` library
- Python packages: `transformers`, `torch`, `librosa`, `safetensors`

## Files

| File | Description |
|------|-------------|
| `mblt-whisper.py` | HuggingFace-compatible Mobilint Whisper model implementation |
| `prepare_model.py` | Prepares model folder with configuration files for inference |
| `inference_mxq.py` | Main inference script for audio transcription/translation |

## Quick Start

### Step 1: Prepare Model Folder

First, prepare a model folder containing the compiled MXQ files and necessary configuration:

```bash
python prepare_model.py \
    --encoder_mxq ../../../compilation/transformers/stt/compilation/compiled/whisper-small_encoder.mxq \
    --decoder_mxq ../../../compilation/transformers/stt/compilation/compiled/whisper-small_decoder.mxq \
    --output_folder ./whisper-small-mxq \
    --base_model openai/whisper-small
```

This script:
- Copies the compiled MXQ files to the output folder
- Downloads the processor (tokenizer + feature extractor) from HuggingFace
- Extracts and saves embedding weights from the base model
- Creates proper configuration for the Mobilint Whisper model

### Step 2: Run Inference

Run speech-to-text inference on an audio file:

```bash
python inference_mxq.py \
    --audio /path/to/audio.wav \
    --model_folder ./whisper-small-mxq
```

## Usage Options

### Basic Transcription

```bash
python inference_mxq.py --audio audio.wav --model_folder ./whisper-small-mxq
```

### Specify Language

```bash
python inference_mxq.py --audio audio.wav --model_folder ./whisper-small-mxq --language en
```

### Translation to English

```bash
python inference_mxq.py --audio audio.wav --model_folder ./whisper-small-mxq --task translate
```

### Use Pipeline API

```bash
python inference_mxq.py --audio audio.wav --model_folder ./whisper-small-mxq --use_pipeline
```

## Command Line Arguments

### prepare_model.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--encoder_mxq` | `../../../compilation/transformers/stt/compilation/compiled/whisper-small_encoder.mxq` | Path to compiled encoder MXQ file |
| `--decoder_mxq` | `../../../compilation/transformers/stt/compilation/compiled/whisper-small_decoder.mxq` | Path to compiled decoder MXQ file |
| `--output_folder` | `./whisper-small-mxq` | Output folder for prepared model |
| `--base_model` | `openai/whisper-small` | HuggingFace model ID for base configuration |

### inference_mxq.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--audio` | `../../../compilation/transformers/stt/data/audio_files/en_us_0000.wav` | Path to audio file |
| `--model_folder` | `./whisper-small-mxq` | Path to prepared model folder |
| `--language` | `None` (auto-detect) | Source language code (e.g., `en`, `ko`, `ja`) |
| `--task` | `transcribe` | Task: `transcribe` or `translate` |
| `--use_pipeline` | `False` | Use HuggingFace pipeline API instead of manual inference |

## Supported Languages

Whisper supports 99+ languages. Common language codes:
- `en` - English
- `ko` - Korean
- `ja` - Japanese
- `zh` - Chinese
- `es` - Spanish
- `fr` - French
- `de` - German

## Architecture

The implementation uses HuggingFace's Auto classes for seamless integration:

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
│                      (maccel)                           │
└─────────────────────────────────────────────────────────┘
```

## Notes

- Audio files are automatically resampled to 16kHz
- The model supports audio chunks up to 30 seconds
- For longer audio, the pipeline API automatically handles chunking
