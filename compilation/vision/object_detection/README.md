# Object Detection Model Compilation

This tutorial provides detailed instructions for compiling object detection models using the Mobilint qbcompiler.

In this tutorial, we will use the [YOLO11m](https://docs.ultralytics.com/models/yolo11/) model, which is pretrained on the COCO dataset developed by Ultralytics. This model is an object detection model that can be used to detect objects in images.

## Prerequisites

Before starting, ensure you have the following installed:

- qbcompiler v1.0.0
- HuggingFace account with access to COCO dataset (to use the gated dataset)

Also, you need to install the following packages:

```bash
pip install ultralytics aiohttp aiofiles
```

## Overview

The compilation process consists of three main steps:

1. **Model Preparation**: Download the model and export it to ONNX format
2. **Calibration Dataset Generation**: Create calibration data from COCO dataset
3. **Model Compilation**: Convert the model to `.mxq` format using calibration data

## Step 1: Model Preparation

First, we need to prepare the model. We will use the `ultralytics` library to download the pretrained model and export it to ONNX format.

```bash
yolo export model=yolo11m.pt format=onnx # Export the model to ONNX format
```

After execution, the exported ONNX model is saved as `yolo11m.onnx` in the current directory.

## Step 2: Calibration Dataset Preparation

A calibration dataset is a set of images that represent the typical input distribution of the model. Since the YOLO11m model is trained on the [COCO dataset](https://cocodataset.org/#download), so we need to prepare the calibration dataset.

Before using the dataset, sign up for an account on [HuggingFace](https://huggingface.co/). Then, log in to HuggingFace using the following command and replace <your_huggingface_token> with your actual HuggingFace token:

```bash
hf auth login --token <your_huggingface_token>
```

If you are not sure about your HuggingFace token, you can find it in your [HuggingFace account settings](https://huggingface.co/settings/tokens).

You may download the files using `prepare_coco.py`, which automatically read the list of available files listed in the dataset, proceed random selection and download files into `coco-selected` directory.

```bash
python prepare_coco.py
```

**What it does:**

- Downloads the dataset of URLs from HuggingFace
- Randomly select images and construct the calibration dataset
- Save the calibration dataset into `coco-selected` directory

**Output:**

- `coco-selected`: Calibration dataset

The selected image dataset is the calibration dataset we will use.

## Step 3: Model Compilation

Before running the model compilation code, you need to verify the preprocessing steps required for calibration. The preprocessing operation of the YOLO model is available on [Ultralytics GitHub](https://github.com/ultralytics/ultralytics). he preprocessing operations that the model use is defined as `LetterBox` operation. 

After the calibration dataset and the model are prepared, we can compile the model.

We designed the code to perform preprocessing inside the compilation API and fuse some operations into the MXQ model to maximize NPU usage.

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
    uint8_input=Uint8InputConfig(apply=True, inputs=[]),
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
python model_compile.py --onnx_path {path_to_onnx_model} --calib_data_path {path_to_calibration_dataset} --save_path {path_to_save_model}
```

**What this does:**

- Loads the ONNX model
- Loads the calibration data
- Compiles the model to `.mxq` format

**Parameters:**

- `--onnx_path`: Path to the ONNX model
- `--calib_data_path`: Path to the calibration data
- `--save_path`: Path to save the MXQ model

**Output:**

- `{path_to_save_model}` file path containing the compiled model

The example command is as follows:

```bash
python model_compile.py --onnx_path ./yolo11m.onnx --calib_data_path ./coco-selected --save_path ./yolo11m.mxq
```

After executing the above command, the compiled model will be saved as `yolo11m.mxq` in the current directory.
