# Bidirectional Encoder Representations from Transformers (BERT)

This tutorial explains how to compile a BERT model with Mobilint `qbcompiler`. The workflow converts a standard BERT model into an optimized `.mxq` file for Mobilint NPUs.

This tutorial uses [Sentence-BERT](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors), a BERT-based model adapted for sentence embedding generation.

## Overview

The workflow consists of four main steps:

1. **Embedding Weight Extraction**: Extract unsupported embedding layers as CPU-side weights
2. **Calibration Data Generation**: Create calibration data for quantization
3. **MBLT Compilation**: Compile the model to MBLT (Mobilint Binary LayouT) format
4. **MXQ Compilation**: Apply quantization and generate the final `.mxq` file

All scripts are run from the `bert/` directory.

## Prerequisites

- Mobilint `qbcompiler` (`>= 1.0.0`)
- GPU with CUDA support (recommended for reducing compilation time)

```bash
pip install -r requirements.txt
```

## Step 1: Extract Embedding Weights

Because of the BERT architecture, some input embedding layers cannot run on the NPU. In this step, you extract those weights from the model and save them as a `.pth` file for CPU-side execution.

```bash
python get_embedding.py
```

**What this does:**

- Loads the Sentence-BERT model from Hugging Face
- Extracts word, token type, and position embeddings, along with LayerNorm weights
- Saves them as a weight dictionary

**Output:**

- `./weights/weight_dict.pth` - Extracted embedding weights

> **Tip:** After Step 3, you can inspect the compiled model in [Netron](https://netron.mobilint.com) to see which layers run on the NPU and which are offloaded to the CPU.

## Step 2: Generate Calibration Data

Generate calibration data from the [STS Benchmark Dataset](https://huggingface.co/datasets/mteb/stsbenchmark-sts). This data is required for quantization during MXQ compilation.

```bash
python prepare_calib.py
```

**What this does:**

- Loads sentences from the STS Benchmark validation split
- Tokenizes and embeds them using the embedding weights extracted in Step 1
- Saves embedded text as NumPy files for calibration

**Output:**

- `./calibration_data/` - Directory containing calibration `.npy` files

## Step 3: Compile to MBLT

Compile the BERT model to the intermediate MBLT (Mobilint Binary LayouT) format.

```bash
# ARIES (default)
python compile_mblt.py

# REGULUS (customers from 2026-06)
python compile_mblt.py --target-device regulus-rb
```

`compile_mblt.py` calls `mblt_compile()` and selects the target NPU with `--target-device` (default: `aries-rb`).

**What this does:**

- Loads the Sentence-BERT model from Hugging Face
- Sets the sequence-length dimension to dynamic
- Configures the attention mask as a padding mask
- Compiles to MBLT format with CPU offload for unsupported layers

**Output:**

- `./mblt/stsb-bert-tiny-safetensors.mblt` - Intermediate MBLT format

## Step 4: Compile to MXQ

Using the calibration data generated in Step 2, compile the model to the final quantized `.mxq` format.

```bash
# ARIES (default)
python compile_mxq.py

# REGULUS (customers from 2026-06)
python compile_mxq.py --target-device regulus-rb
```

`compile_mxq.py` selects the target NPU with `--target-device` (default: `aries-rb`). REGULUS supports only `inference_scheme="single"`, which is set automatically when a `regulus` device is selected.

**What this does:**

- Loads the Sentence-BERT model from Hugging Face
- Applies `CalibrationConfig` with MaxPercentile quantization:
  - Method: WChAMulti (weight per-channel, activation multi-layer)
  - Output: per-layer quantization
  - Percentile: 0.999, Top-k ratio: 0.01
- Compiles the model to `.mxq` format using the calibration data from Step 2

**Output:**

- `./mxq/stsb-bert-tiny-safetensors.mxq` - Final quantized model for NPU

### Target device (`--target-device`)

| User | `--target-device` |
| --- | --- |
| ARIES | `aries-rb` (default) |
| REGULUS (customers from 2026-06) | `regulus-rb` |

> **Note:** BERT compilation is supported on newer REGULUS (`regulus-rb`, for customers from 2026-06). Older REGULUS (`regulus-ra`, for customers before 2026-06) does not support this workflow. Use `--target-device regulus-rb` with both `compile_mblt.py` and `compile_mxq.py`.

## File Structure

```text
bert/
├── get_embedding.py
├── prepare_calib.py
├── compile_mblt.py
├── compile_mxq.py
├── requirements.txt
├── README.md
├── README.KR.md
├── weights/                               # Extracted embedding weights
│   └── weight_dict.pth
├── calibration_data/                      # Calibration data
│   └── *.npy
├── mblt/                                  # Intermediate MBLT model
│   └── stsb-bert-tiny-safetensors.mblt
└── mxq/                                   # Output MXQ model
    └── stsb-bert-tiny-safetensors.mxq
```

## Troubleshooting

### Missing Embedding Weights

If calibration fails because the embedding weights are missing:

```bash
ls ./weights/weight_dict.pth
```

If the file is missing, run `get_embedding.py` again.

### Missing Calibration Data

If MXQ compilation fails because the calibration data is missing:

```bash
ls ./calibration_data/
```

If the directory is missing or empty, run `prepare_calib.py` again.

## References

- [Sentence-BERT](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors)
- [STS Benchmark Dataset](https://huggingface.co/datasets/mteb/stsbenchmark-sts)
- [Mobilint Documentation](https://docs.mobilint.com)
