# Speech-to-Text (STT) Model Compilation

This tutorial provides detailed instructions for compiling the Whisper speech-to-text model using the Mobilint qbcompiler compiler. The compilation process converts the Whisper model (encoder and decoder separately) into optimized `.mxq` format that can run efficiently on Mobilint NPU hardware.

In this tutorial, we will use the [Whisper Small](https://huggingface.co/openai/whisper-small) model, a multilingual speech recognition model developed by OpenAI.

## Overview

The compilation process consists of three main steps:

1. **Data Preparation**: Download multilingual audio data from FLEURS dataset
2. **Calibration Data Generation**: Create calibration datasets for encoder and decoder
3. **Model Compilation**: Compile encoder and decoder separately to `.mxq` format

All scripts are run from the `stt/` directory.

## Prerequisites

```bash
pip install -r requirements.txt
```

## Step 1: Prepare Audio Data

Download audio data from the Google FLEURS dataset. This data includes multilingual audio samples that will be used for calibration.

```bash
python prepare_audio.py
```

**What this does:**

- Downloads audio samples from Google/FLEURS dataset for 17 languages
- Resamples audio to 16kHz mono WAV format

**Supported languages:**

- Arabic, Mandarin Chinese, German, Greek, English, Spanish, French
- Indonesian, Italian, Japanese, Korean, Portuguese, Russian
- Tamil, Thai, Urdu, Vietnamese

**Output:**

- `./audio_files/` - Directory containing WAV audio files

## Step 2: Generate Calibration Data

Generate calibration data for both the Whisper encoder and decoder. This data is essential for quantization during compilation.

This step internally loads the **Whisper Small** model to generate realistic calibration inputs. The encoder calibration extracts mel spectrogram features from audio, while the decoder calibration uses the model to produce transcriptions/translations and converts them into token embeddings.

> **Note:** This step uses the Whisper Small model for inference to generate decoder calibration data. GPU (CUDA) is automatically detected and used if available, significantly speeding up the data generation process. CPU is also fully supported but takes longer.

```bash
python generate_calibration.py
```

**What this does:**

- Generates calibration data for the Whisper encoder (mel spectrogram features)
- Generates calibration data for the Whisper decoder (encoder hidden states + decoder embeddings)
- Loads Whisper Small model to generate transcriptions and translations on-the-fly
- Randomly mixes transcription (80%) and translation (20%) tasks for diverse calibration

**Output:**

- `./calibration_data/encoder/` - Encoder calibration data
  - `whisper_encoder_cali.txt` - List of calibration file paths
  - `encoder_calib_*.npy` - Individual calibration samples
- `./calibration_data/decoder/` - Decoder calibration data
  - `whisper_decoder_calib.json` - Calibration metadata and paths
  - `sample_*/encoder_hidden_states.npy` - Encoder outputs
  - `sample_*/decoder_hidden_states.npy` - Decoder inputs

## Step 3: Compile Models

Compile both the encoder and decoder to `.mxq` format using the calibration data.

### Compile Encoder

```bash
python compile_encoder.py
```

- Loads Whisper Small model from HuggingFace
- Compiles encoder to MBLT format, then to `.mxq` using `all` inference scheme

**Output:**

- `./mblt/whisper-small_encoder.mblt` - Intermediate MBLT format
- `./mxq/whisper-small_encoder.mxq` - Final quantized model for NPU

### Compile Decoder

```bash
python compile_decoder.py
```

- Loads Whisper Small model from HuggingFace
- Compiles decoder to MBLT format, then to `.mxq` with `LlmConfig`

**Output:**

- `./mblt/whisper-small_decoder.mblt` - Intermediate MBLT format
- `./mxq/whisper-small_decoder.mxq` - Final quantized model for NPU

## Troubleshooting

### Out of Memory Errors

- If using GPU, ensure sufficient VRAM (8GB+ recommended)
- Close other GPU-intensive applications
- Reduce the number of calibration samples if needed
- Alternatively, run calibration data generation on CPU (automatic fallback)

### Missing Calibration Data

If compilation fails due to missing calibration data:

```bash
ls ./calibration_data/encoder/whisper_encoder_cali.txt
ls ./calibration_data/decoder/whisper_decoder_calib.json
```

If files are missing, re-run `generate_calibration.py`.

### Audio Download Issues

- Ensure stable internet connection for FLEURS dataset download
- The download script requires access to HuggingFace datasets

## File Structure

```text
stt/
├── prepare_audio.py
├── generate_calibration.py
├── compile_encoder.py
├── compile_decoder.py
├── README.md
├── README.KR.md
├── audio_files/                            # Downloaded audio samples
├── calibration_data/                       # Calibration data
│   ├── encoder/
│   │   ├── whisper_encoder_cali.txt
│   │   └── encoder_calib_*.npy
│   └── decoder/
│       ├── whisper_decoder_calib.json
│       └── sample_*/
│           ├── encoder_hidden_states.npy
│           └── decoder_hidden_states.npy
├── mblt/                                   # Intermediate MBLT models
│   ├── whisper-small_encoder.mblt
│   └── whisper-small_decoder.mblt
└── mxq/                                    # Output MXQ models
    ├── whisper-small_encoder.mxq
    └── whisper-small_decoder.mxq
```

## References

- [OpenAI Whisper](https://github.com/openai/whisper)
- [HuggingFace Whisper](https://huggingface.co/openai/whisper-small)
- [Google FLEURS Dataset](https://huggingface.co/datasets/google/fleurs)
- [Mobilint Documentation](https://docs.mobilint.com)
