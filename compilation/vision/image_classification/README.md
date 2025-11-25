# Image Classification Model Compilation

This tutorial provides detailed instructions for compiling image classification models using the Mobilint qubee compiler.

In this tutorial, we will use the [ResNet-50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) model, which is pretrained on the ImageNet dataset developed by PyTorch. This model is a simple image classification model that can be used to classify images into 1000 classes.

## Prerequisites

Before starting, ensure you have the following installed:

- qubee SDK compiler installed (version >= 0.11 required)
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

A calibration dataset is a set of images that represent the typical input distribution of the model. We will use the [ImageNet dataset](https://huggingface.co/datasets/ILSVRC/imagenet-1k) for this tutorial.

Before using the dataset, sign up for an account on [HuggingFace](https://huggingface.co/) and accept the agreement to use the dataset on the [dataset page](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

Then, log in to HuggingFace using the following command and replace <your_huggingface_token> with your actual HuggingFace token:

```bash
hf auth login --token <your_huggingface_token>
```

If you are not sure about your HuggingFace token, you can find it in your [HuggingFace account settings](https://huggingface.co/settings/tokens).

Then, download the dataset from HuggingFace and save it to the `imagenet-1k-selected` directory. This script will select 1000 images from each class of the dataset and save them to the `imagenet-1k-selected` directory.

```bash
python prepare_imagenet.py
```

**What it does:**

- Downloads the dataset from HuggingFace
- Selects 1000 images from each class of the dataset
- Saves the selected images to the `imagenet-1k-selected` directory

**Output:**

- `imagenet-1k-selected/` directory containing the selected images

With the selected images, we can generate the calibration dataset. Before generating the calibration dataset, you need to confirm the preprocessing that the original model uses. The preprocessing information can be found on the original [ResNet-50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) page. The preprocessing operations that the model uses are: resizing the image to match the shortest side to 256 pixels with bilinear interpolation, center cropping to 224x224 pixels, rescaling the image to the range [0, 1], and normalizing the image with mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225].

The qubee compiler's utility function `make_calib` is designed to generate a calibration dataset with this kind of standard preprocessing operation. The script `prepare_calib.py` uses this function to create a calibration dataset with the preprocessing operation defined above by reading the preprocessing configuration file `resnet50.yaml` and the original calibration data from the `imagenet-1k-selected` directory.

```bash
python prepare_calib.py
```

**What it does:**

- Reads the preprocessing configuration file `resnet50.yaml`
- Reads the original calibration data from the `imagenet-1k-selected` directory
- Generates a calibration dataset with the preprocessing operation defined above
- Saves the calibration dataset to the `resnet50_cali` directory

**Output:**

- `resnet50_cali/` directory containing the calibration dataset
- `resnet50_cali.txt` file containing the paths of the calibration dataset

The calibration dataset is saved to the `resnet50_cali` directory. You can also see the paths recorded in the `resnet50_cali.txt` file.

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
- `--inference_scheme`: Inference scheme(single, multi, global, global4, global8)

**Output:**

- `{path_to_save_model}` file path containing the compiled model

For example, the command is as follows:

```bash
python model_compile.py --onnx_path ./resnet50.onnx --calib_data_path ./resnet50_cali --save_path ./resnet50.mxq --quant_percentile 0.999 --topk_ratio 0.01 --inference_scheme single
```

After executing the above command, the compiled model will be saved as `resnet50.mxq` in the current directory.
