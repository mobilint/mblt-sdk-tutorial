# Object Detection Model Compilation

This tutorial provides comprehensive instructions for compiling object detection models using the Mobilint `qbcompiler`.

We will use the [YOLO11m](https://docs.ultralytics.com/models/yolo11/) model, pretrained on the COCO dataset by Ultralytics. This model is designed for object detection, capable of identifying and localizing multiple objects within an image.

## Prerequisites

Before starting, ensure you have the following installed:

- qbcompiler
- HuggingFace account with access to the COCO dataset (to use the gated dataset)

Also, you need to install the following packages:

```bash
pip install ultralytics aiohttp aiofiles
```

## Overview

The compilation workflow follows three primary steps:

1. **Model Preparation**: Download the model and export it to ONNX format.
2. **Calibration Dataset Preparation**: Create a representative calibration dataset from COCO.
3. **Model Compilation**: Convert the model to the `.mxq` format using the calibration data.

## Step 1: Model Preparation

First, we need to prepare the model. We will use the `ultralytics` library to download the pretrained model and export it to ONNX format.

```bash
yolo export model=yolo11m.pt format=onnx # Export the model to ONNX format
```

After execution, the exported ONNX model is saved as `yolo11m.onnx` in the current directory.

## Step 2: Calibration Dataset Preparation

The calibration dataset consists of images that represent the model's typical input distribution. Since YOLO11m is trained on the [COCO dataset](https://cocodataset.org/#download), we will use COCO samples for calibration.

Before using the dataset, sign up for an account on [HuggingFace](https://huggingface.co/). Then, log in to HuggingFace using the following command and replace <your_huggingface_token> with your actual HuggingFace token:

```bash
hf auth login --token <your_huggingface_token>
```

If you are not sure about your HuggingFace token, you can find it in your [HuggingFace account settings](https://huggingface.co/settings/tokens).

Use the `prepare_coco.py` script to automate the process. This script reads URLs from the COCO dataset, performs a random selection, and downloads the images into the `coco-selected` directory.

```bash
python prepare_coco.py
```

**Action:**

- Downloads COCO image URLs from HuggingFace.
- Randomly selects images to construct the calibration dataset.
- Saves the images to the `coco-selected` directory.

**Output:**

- `coco-selected`: Calibration dataset

The selected image dataset is the calibration dataset we will use.

## Step 3: Model Compilation

Before running the compilation, verify the required preprocessing steps. YOLO models typically use the `LetterBox` operation, as detailed on the [Ultralytics GitHub](https://github.com/ultralytics/ultralytics).

The Mobilint compilation API uses the preprocessing pipeline during calibration. The normalization operation (/255 scaling) is fused into the MXQ model via `Uint8InputConfig`, allowing the model to accept uint8 input directly at runtime. Spatial transforms such as letterbox are not fused and must be performed at runtime.

In `model_compile.py`, we define the preprocessing pipeline as follows. This pipeline is applied to calibration images.

```python
preprocess_pipeline = [
    {
    "op": "letterbox",
    "height": 640,
    "width": 640,
    "padValue": 114
    }
]
preprocessing_config = PreprocessingConfig(
    apply=True,
    auto_convert_format=True,
    pipeline=preprocess_pipeline,
    input_configs={},
)
```

Also, we define the following preprocessing configurations and quantization configuration.

```python
calibration_config = CalibrationConfig(
        method=1,  # 0 for per tensor, 1 for per channel
        output=1,  # 0 for layer, 1 for channel
        mode=1,  # maxpercentile
        max_percentile={
            "percentile": 0.9999,  # quantization percentile
            "topk_ratio": 0.01,  # quantization topk
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

The model differs by device: older REGULUS (`regulus-ra`, customers before 2026-06) supports only YOLOv9 and earlier, so it uses a YOLOv9m model; ARIES (`aries-rb`) and newer REGULUS (`regulus-rb`, customers from 2026-06) use the YOLO11m model.

| User | `--target-device` | Model |
|---|---|---|
| ARIES | `aries-rb` | `yolo11m` |
| REGULUS (customers before 2026-06) | `regulus-ra` | `yolov9m` |
| REGULUS (customers from 2026-06) | `regulus-rb` | `yolo11m` |

Export the matching model first (Step 1 shows `yolo11m`; for `regulus-ra` export `yolov9m` instead):

```bash
# YOLO11 (for aries-rb / regulus-rb)
yolo export model=yolo11m.pt format=onnx
# YOLOv9 (for regulus-ra)
yolo export model=yolov9m.pt format=onnx
```

```bash
# ARIES
python model_compile.py --onnx-path ./yolo11m.onnx --calib-data-path ./coco-selected --save-path ./yolo11m.mxq --mblt-path ./yolo11m.mblt --target-device aries-rb

# REGULUS (customers before 2026-06)
python model_compile.py --onnx-path ./yolov9m.onnx --calib-data-path ./coco-selected --save-path ./yolov9m.mxq --mblt-path ./yolov9m.mblt --target-device regulus-ra

# REGULUS (customers from 2026-06)
python model_compile.py --onnx-path ./yolo11m.onnx --calib-data-path ./coco-selected --save-path ./yolo11m.mxq --mblt-path ./yolo11m.mblt --target-device regulus-rb
```

After executing a command, the corresponding MXQ and MBLT are saved in the current directory.
