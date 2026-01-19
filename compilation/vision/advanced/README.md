# Advanced Test with Image Classification Model

This tutorial provides detailed instructions for compiling image classification models using the Mobilint Qubee compiler.

In this tutorial, we will use the [ResNet-50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) model, which is pretrained on the ImageNet dataset provided by PyTorch. This model is a simple image classification model that can be used to classify images into 1000 classes.

## Prerequisites

Before starting, ensure you have the following installed:

- Qubee SDK compiler installed (version >= 0.11 required)
- HuggingFace account with access to ImageNet dataset (if using gated dataset)

## Overview

The compilation process consists of three main steps:

1. **Model Preparation**: Download the model and export it to ONNX format
2. **Calibration Dataset Generation**: Create calibration data from ImageNet dataset
3. **Model Compilation**: Convert the model to `.mxq` format using calibration data

Also, you need to install the following packages:

```bash
pip install datasets
```

## Step 1: Model Preparation

First, we need to prepare the model. We will use the `torchvision` library to download the pretrained model and export it to ONNX format through `torch.onnx.export`.

```python
import torch
from torchvision.models import resnet50, ResNet50_Weights

# Using pretrained weights:
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()
# make dummy input depending on the model's input shape
input = torch.randn(1, 3, 224, 224)
# export to onnx
torch.onnx.export(model, input, "resnet50.onnx")
```

By executing the above code (`prepare_model.py`), the exported ONNX model is saved as `resnet50.onnx` in the current directory.

## Step 2: Calibration Dataset Preparation

To prepare the calibration dataset, run the following command to get the original source image sets.

```bash
python prepare_imagenet.py
```

This code will create 5 folders:

- `imagenet-1k-100cls-100`
- `imagenet-1k-10cls-10`
- `imagenet-1k-20cls-100`
- `imagenet-1k-5cls-100`
- `imagenet-1k-1000cls-1000`

Each folder contains images for specific classes. From each dataset, you may generate calibration data by running the following command:

```bash
python prepare_calib.py --data_dir {path_to_imagenet_dataset}
```

## Step 3: Model Compilation

After the calibration dataset and the model are prepared, we can compile the model.

```bash
python model_compile.py --onnx_path {path_to_onnx_model} --calib_data_path {path_to_calibration_dataset} --save_path {path_to_save_model} --quant_percentile {quantization_percentile} --topk_ratio {topk_ratio} --inference_scheme {inference_scheme}
```

**What it does:**

- Loads the ONNX model
- Loads the calibration data
- Compiles the model to `.mxq` format

**Parameters:**

- `--onnx_path`: Path to the ONNX model
- `--calib_data_path`: Path to the calibration data
- `--save_path`: Path to save the MXQ model
- `--quant_percentile`: Quantization percentile
- `--topk_ratio`: Top-k ratio
- `--inference_scheme`: Inference scheme (single, multi, global, global4, global8)

**Output:**

- `{path_to_save_model}` file path containing the compiled model

For example, the command is as follows:

```bash
python model_compile.py --onnx_path ./resnet50.onnx --calib_data_path ./resnet50_imagenet-1k-5cls-100 --save_path ./resnet50_5cls_100_9999_01.mxq --quant_percentile 0.9999 --topk_ratio 0.01 --inference_scheme single
```

After executing the above command, the compiled model will be saved as `resnet50_5cls_100_9999_01.mxq` in the current directory.
