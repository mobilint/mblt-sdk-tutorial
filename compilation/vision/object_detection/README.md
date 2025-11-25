# Object Detection Model Compilation

This tutorial provides detailed instructions for compiling object detection models using the Mobilint qb compiler.

In this tutorial, we will use the [YOLO11m](https://docs.ultralytics.com/models/yolo11/) model, which is pretrained on the COCO dataset developed by Ultralytics. This model is an object detection model that can be used to detect objects in images.

## Prerequisites

Before starting, ensure you have the following installed:

- qubee SDK compiler installed (version >= 0.11 required)

Also, you need to install the following packages:

```bash
pip install ultralytics
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

The YOLO11m model is trained on the COCO dataset, so we need to prepare the calibration dataset.

```bash
wget http://images.cocodataset.org/zips/val2017.zip # Download the validation dataset
unzip val2017.zip # Unzip the dataset
```

> Note: According to the [COCO dataset](https://cocodataset.org/#download) page, downloading the dataset through Google Cloud Platform is recommended, but currently it is not available.

The calibration dataset should be pre-processed to be compatible with the quantized model. Therefore, we should first investigate the pre-processing operation used in the original model. The pre-processing operation is defined in [Ultralytics' GitHub](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py). We wrote the simplified but equivalent operation as follows:

```python
import numpy as np
import cv2

img_size = [640, 640] 
def preprocess_yolo(img_path: str):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h0, w0 = img.shape[:2]  # original hw
    r = min(img_size[0] / h0, img_size[1] / w0)  # ratio
    new_unpad = int(round(w0 * r)), int(round(h0 * r))
    dh, dw = (
        img_size[0] - new_unpad[1],
        img_size[1] - new_unpad[0],
    )  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if (img.shape[1], img.shape[0]) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # add border
    img = (img / 255).astype(np.float32)

    return img

```

One of the qb compiler's utility functions is `make_calib_man`, which can be used to create a calibration dataset with custom pre-processing functions. The script `prepare_calib.py` uses this function to create a calibration dataset with the pre-processing operation defined above.

```bash
python3 prepare_calib.py --data_dir {path_to_calibration_dataset} --img_size {image_size} --save_dir {path_to_save_calibration_dataset} --save_name {name_of_calibration_dataset} --max_size {maximum_number_of_calibration_data}
```

**What this does:**

- Loads the COCO dataset
- Pre-processes the images using the pre-processing operation defined above
- Saves the pre-processed images as calibration data

**Parameters:**

- `--data_dir`: Path to the calibration dataset
- `--img_size`: Image size
- `--save_dir`: Path to save the calibration dataset
- `--save_name`: Name of the calibration dataset
- `--max_size`: Maximum number of calibration data

**Output Location:**
The calibration dataset will be saved in the directory specified by `--save_dir`.

The example command is as follows:

```bash
python3 prepare_calib.py --data_dir ./val2017 --img_size 640 --save_dir ./ --save_name yolo11m_cali --max_size 100
```

## Step 3: Model Compilation

After the calibration dataset and the model are prepared, we can compile the model.

```bash
python3 model_compile.py --onnx_path {path_to_onnx_model} --calib_data_path {path_to_calibration_dataset} --save_path {path_to_save_model} --quant_percentile {quantization_percentile} --topk_ratio {topk_ratio} --inference_scheme {inference_scheme}
```

**What this does:**

- Loads the ONNX model
- Loads the calibration data
- Compiles the model to `.mxq` format

**Parameters:**

- `--onnx_path`: Path to the ONNX model
- `--calib_data_path`: Path to the calibration data
- `--save_path`: Path to save the MXQ model
- `--quant_percentile`: Quantization percentile
- `--topk_ratio`: Top-k ratio
- `--inference_scheme`: Inference scheme

**Output Location:**
The compiled model will be saved in the directory specified by `--save_path`.

The quantization percentile and top-k ratio are parameters that are required for running quantization algorithm.

The inference scheme is a parameter that specifies the core allocation strategy for the model. Currently, the following inference schemes are supported:

- single: Single core inference
- multi: Multi-core inference
- global: Global inference (Deprecated and replaced by global8)
- global4: Global inference with 4 cores
- global8: Global inference with 8 cores

Further details about the inference scheme can be found in the [Multi-Core Modes](https://docs.mobilint.com/v0.29/en/multicore.html) documentation.

The example command is as follows:

```bash
python3 model_compile.py --onnx_path ./yolo11m.onnx --calib_data_path ./yolo11m_cali --save_path ./yolo11m.mxq --quant_percentile 0.999 --topk_ratio 0.001 --inference_scheme single
```

After executing the above command, the compiled model will be saved as `yolo11m.mxq` in the current directory.