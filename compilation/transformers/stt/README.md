# Speech-to-Text (STT) Model Compilation

This tutorial provides detailed instructions for compiling the Whisper speech-to-text model using the Mobilint qbcompiler compiler. The compilation process converts the Whisper model (encoder and decoder separately) into optimized `.mxq` format that can run efficiently on Mobilint NPU hardware.

In this tutorial, we will use the [Whisper Small](https://huggingface.co/openai/whisper-small) model, a multilingual speech recognition model developed by OpenAI.

## Prerequisites

Before starting, ensure you have the following:

- Python 3.8 or higher
- GPU with CUDA support (required for calibration and compilation)
- Sufficient disk space (~10GB for model + calibration data)

### Install qbcompiler Compiler

Download and install the qbcompiler compiler (version 1.0.1) from the Mobilint release page:

1. Go to [https://dl.mobilint.com/releases?series-id=1](https://dl.mobilint.com/releases?series-id=1)
2. Download the appropriate `.whl` file for qbcompiler version **1.0.1**
3. Install the wheel file:

```bash
pip install qbcompiler-1.0.1-<your-platform>.whl
```

## Overview

The compilation process consists of three main steps:

1. **Data Download**: Download multilingual audio data from FLEURS dataset
2. **Calibration Data Generation**: Create calibration datasets for encoder and decoder
3. **Model Compilation**: Compile encoder and decoder separately to `.mxq` format

## Step 1: Download Data

First, download audio data from the Google FLEURS dataset. This data includes multilingual audio samples that will be used for calibration.

### Install Dependencies

```bash
cd data
```

**Required packages:**
- datasets
- librosa
- soundfile
- openai-whisper

### Download Audio Data

```bash
python download_data.py
```

**What this does:**
- Downloads audio samples from Google/FLEURS dataset for 17 languages
- Loads Whisper Large V3 for generating transcriptions and translations
- Saves audio files resampled to 16kHz mono WAV format
- Generates transcriptions using Whisper with source language specified
- Generates English translations using Whisper's translate task

**Supported languages:**
- Arabic, Mandarin Chinese, German, Greek, English, Spanish, French
- Indonesian, Italian, Japanese, Korean, Portuguese, Russian
- Tamil, Thai, Urdu, Vietnamese

**Output:**
- `audio_files/` - Directory containing WAV audio files
- `transcriptions.json` - Whisper transcriptions for each audio file
- `translations.json` - English translations with metadata

## Step 2: Create Calibration Data

Generate calibration data for both the Whisper encoder and decoder. This data is essential for quantization during compilation.

### Install Dependencies

```bash
cd ../calibration
```

**Required packages:**
- torch==2.7.1
- transformers==4.50.0
- librosa

### Generate Calibration Data

```bash
python create_calibration.py
```

**What this does:**
- Creates calibration data for the Whisper encoder (mel spectrogram features)
- Creates calibration data for the Whisper decoder (encoder hidden states + decoder embeddings)
- Generates transcriptions and translations on-the-fly using whisper-small
- Supports both transcription and translation tasks with proper special tokens
- Randomly mixes transcription (80%) and translation (20%) tasks for diverse calibration

**Encoder Calibration:**
- Processes audio files through HuggingFace Whisper processor
- Extracts mel spectrogram features with shape `[1, 80, 3000]` (transposed to `[1, 3000, 80]` for saving)
- Saves calibration files as `.npy` format

**Decoder Calibration:**
- Processes audio through encoder to get hidden states
- Generates transcriptions and translations using whisper-small model
- Applies text normalization using HuggingFace processor
- Generates token sequences with proper special tokens (SOT, language, task, timestamps)
- Creates input embeddings (token embeddings + positional embeddings) for decoder calibration
- Saves both encoder and decoder hidden states

**Output:**
- `encoder/` - Encoder calibration data
  - `whisper_encoder_cali.txt` - List of calibration file paths
  - `encoder_calib_*.npy` - Individual calibration samples
- `decoder/` - Decoder calibration data
  - `whisper_decoder_calib.json` - Calibration metadata and paths
  - `whisper_decoder_calib_metadata.json` - Decoded token sequences for reference
  - `sample_*/encoder_hidden_states.npy` - Encoder outputs
  - `sample_*/decoder_hidden_states.npy` - Decoder inputs

## Step 3: Compile Models

Compile both the encoder and decoder to `.mxq` format using the calibration data.

### Install Dependencies

```bash
cd ../compilation
```

**Required packages:**
- transformers==4.50.0
- qbcompiler==1.0.1 (with modification described in Prerequisites)

### Compile Encoder

```bash
python compile_encoder.py
```

**What this does:**
- Loads Whisper Small model from HuggingFace
- Compiles encoder to MBLT format first
- Uses encoder calibration data for quantization
- Compiles to final `.mxq` format using `global4` inference scheme

**Output:**
- `compiled/whisper-small_encoder.mblt` - Intermediate MBLT format
- `compiled/whisper-small_encoder.mxq` - Final quantized model for NPU

### Compile Decoder

> **Important:** Before running decoder compilation, ensure you have applied the modification to qbcompiler's parser.py as described in the Prerequisites section.

```bash
python compile_decoder.py
```

**What this does:**
- Loads Whisper Small model from HuggingFace
- Compiles decoder to MBLT format first
- Uses decoder calibration data with full sequence length calibration (`use_full_seq_len_calib=True`)
- Applies LLM-specific configurations via `get_llm_config()` for decoder compilation
- Compiles to final `.mxq` format

**Output:**
- `compiled/whisper-small_decoder.mblt` - Intermediate MBLT format
- `compiled/whisper-small_decoder.mxq` - Final quantized model for NPU

## Complete Compilation Pipeline

Here's the complete sequence of commands:

```bash
# Step 1: Download audio data
cd data
python download_data.py

# Step 2: Generate calibration data
cd ../calibration
python create_calibration.py

# Step 3: Compile models
cd ../compilation

# Compile encoder
python compile_encoder.py

# Compile decoder (requires qbcompiler modification - see Prerequisites)
python compile_decoder.py
```

## Output Summary

After completing all steps, you will have:

### Data Files
- `data/audio_files/` - Audio samples from FLEURS dataset
- `data/transcriptions.json` - Whisper transcriptions
- `data/translations.json` - English translations

### Calibration Files
- `calibration/encoder/` - Encoder calibration data
- `calibration/decoder/` - Decoder calibration data

### Compiled Models
- `compilation/compiled/whisper-small_encoder.mxq` - Quantized encoder for NPU
- `compilation/compiled/whisper-small_decoder.mxq` - Quantized decoder for NPU

## Troubleshooting

### Out of Memory Errors

- Ensure you have a GPU with sufficient VRAM (8GB+ recommended)
- Close other GPU-intensive applications
- Reduce the number of calibration samples if needed

### Missing Calibration Data

If compilation fails due to missing calibration data:

```bash
# Check that calibration files exist
ls ../calibration/encoder/whisper_encoder_cali.txt
ls ../calibration/decoder/whisper_decoder_calib.json
```

If files are missing, re-run `create_calibration.py` from the calibration folder.

### Audio Download Issues

- Ensure stable internet connection for FLEURS dataset download
- The download script requires access to HuggingFace datasets

## File Structure

```
stt/
├── README.md
├── data/
│   ├── download_data.py
│   ├── audio_files/                        # Downloaded audio samples
│   ├── transcriptions.json                 # Generated transcriptions
│   └── translations.json                   # Generated translations
├── calibration/
│   ├── create_calibration.py
│   ├── encoder/                            # Encoder calibration data
│   │   ├── whisper_encoder_cali.txt        # Calibration file list
│   │   └── encoder_calib_*.npy             # Mel spectrogram features
│   └── decoder/                            # Decoder calibration data
│       ├── whisper_decoder_calib.json      # Calibration config and paths
│       ├── whisper_decoder_calib_metadata.json  # Decoded tokens
│       └── sample_*/                       # Per-sample data
│           ├── encoder_hidden_states.npy
│           └── decoder_hidden_states.npy
└── compilation/
    ├── compile_encoder.py
    ├── compile_decoder.py
    └── compiled/                           # Output MXQ models
        ├── whisper-small_encoder.mblt
        ├── whisper-small_encoder.mxq
        ├── whisper-small_decoder.mblt
        └── whisper-small_decoder.mxq
```

## References

- [OpenAI Whisper](https://github.com/openai/whisper)
- [HuggingFace Whisper](https://huggingface.co/openai/whisper-small)
- [Google FLEURS Dataset](https://huggingface.co/datasets/google/fleurs)
- [Mobilint Documentation](https://docs.mobilint.com)

## Support

For issues or questions:
- Check the troubleshooting section above
- Review qbcompiler SDK documentation
- Contact Mobilint support with detailed error logs
