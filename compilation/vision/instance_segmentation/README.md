# Instance Segmentation Model Compilation

This tutorial provides comprehensive instructions for compiling instance segmentation models using the Mobilint `qbcompiler`.

We will use the [YOLO11m-seg](https://docs.ultralytics.com/models/yolo11/) model, pretrained on the COCO dataset by Ultralytics. This model performs instance segmentation, identifying and masking individual objects within an image.

## Prerequisites

Before starting, ensure you have the following installed:

- qbcompiler v1.0.0
- HuggingFace account with access to COCO dataset (to use the gated dataset)

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
yolo export model=yolo11m-seg.pt format=onnx # Export the model to ONNX format
```

After execution, the exported ONNX model is saved as `yolo11m-seg.onnx` in the current directory.

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

Before running the compilation, verify the required preprocessing steps. YOLO models typically use the `LetterBox` operation, as detailed on the [Ultralytics GitHub](https://github.com/ultralytics/ultralytics).

The Mobilint compilation API performs these preprocessing steps internally and fuses operations directly into the MXQ model to maximize NPU efficiency.

In `model_compile.py`, we define the preprocessing pipeline as follows. This pipeline is used in calibration and will fuse the normalization module into the deep learning model.

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
input_process_config = InputProcessConfig(
    uint8_input=Uint8InputConfig(apply=True, inputs=[]), # uint8 input
    image_channels=3,
    preprocessing=preprocessing_config,
)

quantization_config = QuantizationConfig.from_kwargs(
    quantization_method=1,  # 0 for per tensor, 1 for per channel
    quantization_output=1,  # 0 for layer, 1 for channel
    quantization_mode=2,  # maxpercentile
    percentile=0.999,
    topk_ratio=0.01,
)
```

After configuring the settings, the code can be executed as follows.

```bash
python model_compile.py --onnx-path {path_to_onnx_model} --calib-data-path {path_to_calibration_dataset} --save-path {path_to_save_model}
```

**What this does:**

- Loads the ONNX model
- Loads the calibration data
- Compiles the model to `.mxq` format

**Parameters:**

- `--onnx-path`: Path to the ONNX model
- `--calib-data-path`: Path to the calibration data
- `--save-path`: Path to save the MXQ model

**Output:**

- `{path_to_save_model}` file path containing the compiled model

The example command is as follows:

```bash
python model_compile.py --onnx-path ./yolo11m-seg.onnx --calib-data-path ./coco-selected --save-path ./yolo11m-seg.mxq 
```

After executing the above command, the compiled model will be saved as `yolo11m-seg.mxq` in the current directory.
