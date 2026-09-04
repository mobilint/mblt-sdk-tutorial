# Vision-Language Model Compilation

This tutorial compiles the encoder and decoder of [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) into MXQ models and prepares one self-contained runtime model directory.

Run all commands from `compilation/vlm`.

## Prerequisites

```bash
pip install -r requirements.txt
```

## Supported Devices

| Device | Support |
| --- | --- |
| `aries-rb` | Supported |
| `regulus-rb` | Supported |
| `regulus-ra` | Not supported |

## 1. Download Calibration Images

```bash
python download_images.py
```

The script downloads 300 COCO validation images from a fixed dataset revision, converts them to RGB, and resizes
them to `224x224` under `./images`.

## 2. Generate Calibration Data

```bash
python generate_calibration_data.py --batch-size 4
```

The script creates vision encoder samples and decoder prefill/decode samples under `./calibration_data`. The default
batch size is 4 on `cuda:0`, which is sized for a 24 GiB GPU. Use `--device` to select another GPU.

```text
calibration_data/
├── vision/
│   └── npy_files.txt
├── prefill/
│   └── npy_files.json
├── decode/
│   └── npy_files.json
└── language/
    └── npy_files.json
```

The dataset revision, random seed, image order, and prompt order are fixed. Repeated runs with the same options,
GPU, and software environment produce identical calibration files. Only generations that reach EOS are included. If
`./calibration_data` already exists, pass `--force` to replace it.

## 3. Compile MXQ Models

Compile the decoder first. Decoder compilation produces the SpinR1 matrix required by encoder compilation and runtime model preparation.

For ARIES:

```bash
python compile_decoder.py --target-device aries-rb
python compile_encoder.py --target-device aries-rb
```

For REGULUS:

```bash
python compile_decoder.py --target-device regulus-rb
python compile_encoder.py --target-device regulus-rb
```

Each script creates its target-specific MBLT and then compiles the MXQ model. Compiler options for both scripts are defined in `compile_config.py`.

```text
mblt/<target-device>/Qwen_Qwen3-VL-2B-Instruct_{decoder,encoder}.mblt
mxq/<target-device>/Qwen3-VL-2B-Instruct_{decoder,encoder}.mxq
spinWeight/<target-device>/global_rotation.pth
```

The validated Qwen3-VL 2B compiler configuration is applied automatically. ARIES uses `inference_scheme="all"`. REGULUS uses `inference_scheme="single"` with a maximum sequence and cache length of 1024.

## 4. Prepare the Runtime Model

Run this after both MXQ files have been compiled.

For ARIES:

```bash
python prepare_model.py --target-device aries-rb
```

For REGULUS:

```bash
python prepare_model.py --target-device regulus-rb
```

The script downloads the Mobilint runtime files, applies the decoder SpinR1 matrix to the token embedding, copies both MXQ files, and writes the matching runtime configuration.

The output is written to `./prepared/<target-device>/Qwen3-VL-2B-Instruct`. If that directory already exists, pass `--force` to replace it.

## Runtime

Continue with the [Python VLM runtime tutorial](../../runtime/python/vlm/README.md).
