# BERT Compilation

This tutorial compiles [Sentence-BERT](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) into an MXQ model for Mobilint NPUs.

Input embeddings and mean pooling run on the CPU, while the BERT encoder runs on the NPU.
The workflow does not rotate or transform the embedding weights.
Calibration and compilation load the original model directly from Hugging Face.
The final preparation step places the original model files and compiled MXQ in one runtime directory.

Run all commands from this directory.

## Prerequisites

```bash
pip install -r requirements.txt
```

## 1. Generate Calibration Data

```bash
python generate_calib.py \
  --model-id sentence-transformers-testing/stsb-bert-tiny-safetensors \
  --output-dir ./calibration_data
```

The script selects 256 sentences from the STS Benchmark validation split, applies the original BERT embedding layer, and saves the results to `./calibration_data`.
The output directory must be empty.

## 2. Compile the Model

```bash
python compile_model.py \
  --model-id sentence-transformers-testing/stsb-bert-tiny-safetensors \
  --calib-data-path ./calibration_data \
  --mblt-path ./mblt/stsb-bert-tiny-safetensors.mblt \
  --save-path ./mxq/stsb-bert-tiny-safetensors.mxq \
  --target-device aries-rb
```

For REGULUS, run:

```bash
python compile_model.py --target-device regulus-rb
```

The MBLT output is saved to `./mblt/stsb-bert-tiny-safetensors.mblt`.
The MXQ output is saved to `./mxq/stsb-bert-tiny-safetensors.mxq`.

### Supported Devices

| Device | Support |
| --- | --- |
| `aries-rb` | Supported |
| `regulus-rb` | Supported |
| `regulus-ra` | Not supported |

## 3. Prepare the Runtime Model

Run this after MXQ compilation.
`prepare_model.py` downloads the original model files and copies the compiled MXQ into one runtime directory.

```bash
python prepare_model.py \
  --model-id sentence-transformers-testing/stsb-bert-tiny-safetensors \
  --mxq-path ./mxq/stsb-bert-tiny-safetensors.mxq \
  --output-dir ./bert-mxq
```

The `./bert-mxq` directory contains the model weights, configuration, tokenizer, pooling configuration, and compiled MXQ required by the runtime examples.

## File Structure

```text
bert/
├── compile_model.py
├── generate_calib.py
├── prepare_model.py
├── requirements.txt
├── README.md
├── README.KR.md
├── calibration_data/
│   └── *.npy
├── mblt/
│   └── stsb-bert-tiny-safetensors.mblt
├── mxq/
│   └── stsb-bert-tiny-safetensors.mxq
└── bert-mxq/
    ├── model.safetensors
    ├── stsb-bert-tiny-safetensors.mxq
    ├── 1_Pooling/config.json
    └── tokenizer and model configuration files
```

## Runtime

After compilation, follow the [BERT Python runtime tutorial](../../runtime/python/bert/README.md).
