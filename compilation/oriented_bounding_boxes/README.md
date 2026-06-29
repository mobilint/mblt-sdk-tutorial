# Oriented Bounding Boxes Model Compilation

This tutorial explains how to compile an oriented bounding box (OBB) detection model with Mobilint `qbcompiler`.

We use the [YOLO11m-obb](https://docs.ultralytics.com/tasks/obb/) model from Ultralytics. This model is trained for rotated object detection and is commonly used with aerial-image datasets such as DOTA.

## Prerequisites

Before starting, ensure that the following are available:

- `qbcompiler`
- Python 3.10 or later

Install the tutorial dependencies:

```bash
pip install ultralytics
```

The dataset preparation script uses only the Python standard library.

## Overview

The compilation workflow follows three steps:

1. **Model Preparation**: Export the pretrained OBB model to ONNX.
2. **Calibration Dataset Preparation**: Download DOTAv1 and select calibration images.
3. **Model Compilation**: Convert the ONNX model into the `.mxq` format.

## Step 1: Model Preparation

Export the pretrained YOLO11 OBB model to ONNX:

```bash
yolo export model=yolo11m-obb.pt format=onnx
```

After the export finishes, the ONNX model is saved as `yolo11m-obb.onnx` in the current directory.

## Step 2: Calibration Dataset Preparation

The calibration dataset should reflect the model's expected input distribution. Because `yolo11m-obb.pt` is trained for DOTA-style aerial imagery, this tutorial uses DOTAv1 as calibration data.

You can prepare the dataset directly from the Ultralytics archive:

```bash
python prepare_dota.py
```

This script:

- Downloads `DOTAv1.zip` if it is not already present.
- Extracts the archive into `./DOTAv1`.
- Randomly selects 100 images with a fixed seed.
- Copies those images into `./dota-selected`.

The resulting `dota-selected` directory is the calibration dataset used by `model_compile.py`.

### Optional Dataset Arguments

If you already downloaded the dataset manually, you can reuse it:

```bash
python prepare_dota.py --skip-download --zip-path ./DOTAv1.zip
```

If you extracted the dataset yourself and only want to create the calibration subset:

```bash
python prepare_dota.py --skip-download --extract-dir ./DOTAv1 --output-dir ./dota-selected --num-images 100
```

When `--skip-download` is set, an existing `--extract-dir` is reused even if it was extracted manually and does not
contain the script's `.extracted` marker file.

## Step 3: Model Compilation

Before compiling, verify the preprocessing required by the exported model. Ultralytics OBB models use letterbox resizing, and this tutorial matches that behavior during calibration.

In `model_compile.py`, the preprocessing pipeline is configured as:

```python
preprocess_pipeline = [{"op": "letterbox", "height": 1024, "width": 1024, "padValue": 114}]
preprocessing_config = PreprocessingConfig(
    apply=True,
    auto_convert_format=True,
    pipeline=preprocess_pipeline,
    input_configs={},
)
```

The Mobilint compilation API applies this pipeline during calibration. The `/255` normalization is fused into the MXQ model through `Uint8InputConfig`, so the runtime can pass `uint8` input directly. Spatial transforms such as letterbox are not fused and must still be applied at runtime.

The quantization settings are defined as:

```python
calibration_config = CalibrationConfig(
    method=1,
    output=1,
    mode=1,
    max_percentile={
        "percentile": 0.9999,
        "topk_ratio": 0.01,
    },
)
```

After configuring the settings, run with `--target-device` for your hardware. A single `model_compile.py` produces both outputs: the quantized MXQ (`--save-path`) and the intermediate MBLT graph (`--mblt-path`).

**Parameters:**

- `--onnx-path`: Path to the ONNX model
- `--calib-data-path`: Path to the calibration data
- `--save-path`: Path to save the MXQ model (onnx -> mxq output)
- `--mblt-path`: Path to save the MBLT intermediate graph (onnx -> mblt output)
- `--target-device` (required): Target NPU. See the table below. The inference scheme is derived automatically (ARIES = `all`, REGULUS = `single`).

**Output:**

- MXQ model at `--save-path` (onnx -> mxq, quantized NPU package)
- MBLT intermediate graph at `--mblt-path` (onnx -> mblt, pre-quantization graph)

### Selecting the target device (`--target-device`)

| User | `--target-device` | Model |
|---|---|---|
| ARIES | `aries-rb` | `yolo11m-obb` |
| REGULUS (customers from 2026-06) | `regulus-rb` | `yolo11m-obb` |

> **Note**: OBB uses the YOLO11 `yolo11m-obb` model, which is **not supported on older REGULUS (`regulus-ra`, customers before 2026-06)** — that generation supports only YOLOv9 and earlier, and OBB is available only on later models. Use `aries-rb` or `regulus-rb`.

```bash
# ARIES
python model_compile.py --onnx-path ./yolo11m-obb.onnx --calib-data-path ./dota-selected --save-path ./yolo11m-obb.mxq --mblt-path ./yolo11m-obb.mblt --target-device aries-rb

# REGULUS (customers from 2026-06)
python model_compile.py --onnx-path ./yolo11m-obb.onnx --calib-data-path ./dota-selected --save-path ./yolo11m-obb.mxq --mblt-path ./yolo11m-obb.mblt --target-device regulus-rb
```

After executing a command, the MXQ (`yolo11m-obb.mxq`) and MBLT (`yolo11m-obb.mblt`) are saved in the current directory.

## Files in This Tutorial

- `model_compile.py`: Compiles the ONNX model into MXQ / MBLT for the selected `--target-device`.
- `prepare_dota.py`: Downloads or reuses DOTAv1 and prepares calibration images.
- `README.md`: Documents the end-to-end workflow for this example.
