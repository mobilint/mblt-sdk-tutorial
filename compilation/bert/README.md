# Bidirectional Encoder Representations from Transformers (BERT)

This tutorial provides detailed instructions for compiling a BERT model using the Mobilint qb Compiler. The compilation process converts a standard BERT model into an optimized `.mxq` format that can run efficiently on Mobilint NPU hardware.

In this tutorial, we will use the [Sentence-BERT](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) model, which is based on the BERT architecture and modified to generate sentence embeddings.

## Overview

The compilation process consists of four main steps:

1. **Embedding Weight Extraction**: Extract unsupported embedding layers as CPU-side weights
2. **Calibration Data Generation**: Create calibration datasets for quantization
3. **MBLT Compilation**: Compile the model to MBLT (Mobilint Binary LayouT) format
4. **MXQ Compilation**: Apply quantization and compile to `.mxq` format for deployment

All scripts are run from the `bert/` directory.

## Prerequisites

- Mobilint qb Compiler (version >= 1.0.0 required)
- GPU with CUDA support (recommended for reducing compilation time)

```bash
pip install -r requirements.txt
```

## Step 1: Extract Embedding Weights

Due to its complex architecture, some input embedding layers of BERT are not supported by the NPU. Therefore, we extract the embedding weights from the model and save them as a `.pth` file for CPU-side execution.

```bash
python get_embedding.py
```

**What this does:**

- Loads the Sentence-BERT model from HuggingFace
- Extracts word, token type, position embeddings and LayerNorm weights
- Saves them as a weight dictionary

**Output:**

- `./weights/weight_dict.pth` - Extracted embedding weights

> **Tip:** You can visualize the model architecture using [Netron](https://netron.mobilint.com) after MBLT compilation (Step 3) to see which layers are supported and which are offloaded to CPU.

## Step 2: Generate Calibration Data

Generate calibration data using the [STS Benchmark Dataset](https://huggingface.co/datasets/mteb/stsbenchmark-sts). This data is essential for quantization during MXQ compilation.

```bash
python prepare_calib.py
```

**What this does:**

- Loads sentences from the STS Benchmark validation set
- Tokenizes and embeds them using the extracted embedding weights (Step 1)
- Saves embedded text as NumPy files for calibration

**Output:**

- `./calibration_data/` - Directory containing calibration `.npy` files

## Step 3: Compile to MBLT

Compile the BERT model to MBLT (Mobilint Binary LayouT) intermediate format.

```bash
python compile_mblt.py
```

**What this does:**

- Loads the Sentence-BERT model from HuggingFace
- Sets sequence length dimension as dynamic
- Configures attention mask as padding mask
- Compiles to MBLT format with CPU offload for unsupported layers

**Output:**

- `./mblt/stsb-bert-tiny-safetensors.mblt` - Intermediate MBLT format

## Step 4: Compile to MXQ

Compile the model to final `.mxq` format with quantization using the calibration data.

```bash
python compile_mxq.py
```

**What this does:**

- Loads the Sentence-BERT model from HuggingFace
- Applies `CalibrationConfig` with MaxPercentile quantization:
  - Method: WChAMulti (weight per-channel, activation multi-layer)
  - Output: per-layer quantization
  - Percentile: 0.999, Top-k ratio: 0.01
- Compiles to `.mxq` format using calibration data from Step 2

**Output:**

- `./mxq/stsb-bert-tiny-safetensors.mxq` - Final quantized model for NPU

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

If calibration fails due to missing weights:

```bash
ls ./weights/weight_dict.pth
```

If the file is missing, re-run `get_embedding.py`.

### Missing Calibration Data

If MXQ compilation fails due to missing calibration data:

```bash
ls ./calibration_data/
```

If the directory is empty or missing, re-run `prepare_calib.py`.

## References

- [Sentence-BERT](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors)
- [STS Benchmark Dataset](https://huggingface.co/datasets/mteb/stsbenchmark-sts)
- [Mobilint Documentation](https://docs.mobilint.com)
