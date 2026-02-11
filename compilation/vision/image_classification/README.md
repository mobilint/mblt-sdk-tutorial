# Image Classification Model Compilation

This tutorial provides comprehensive instructions for compiling image classification models using the Mobilint `qbcompiler`.

We will use the [ResNet-50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) model, available via `torchvision`. This model, pretrained on the ImageNet-1K dataset, is a standard benchmark for classifying images into 1,000 distinct categories.

## Prerequisites

Before starting, ensure you have the following installed:

- qbcompiler v1.0.0
- HuggingFace account with access to ImageNet dataset (to use the gated dataset)

## Overview

The compilation workflow consists of three primary steps:

1. **Model Preparation**: Download the model and export it to ONNX format.
2. **Calibration Dataset Preparation**: Create a representative calibration dataset from ImageNet.
3. **Model Compilation**: Convert the model to the `.mxq` format using the calibration data.

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

# Create dummy input based on the model's input shape
input = torch.randn(1, 3, 224, 224)

# Export to ONNX
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

Then, download the dataset from HuggingFace and save it to the `imagenet-1k-selected` directory. This script will select 1 images from each class of the dataset and save 1000 image files to the `imagenet-1k-selected` directory.

```bash
python prepare_imagenet.py
```

**What it does:**

- Downloads the dataset from HuggingFace
- Selects 1 image from each class of the dataset
- Saves the selected images to the `imagenet-1k-selected` directory

**Output:**

- `imagenet-1k-selected/` directory containing the selected images

The selected image dataset is the calibration dataset we will use.

## Step 3: Model Compilation

Before running the compilation, verify the preprocessing steps required by the model. According to the [official ResNet-50 documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html), the required operations include:
- Resizing the shorter side to 256 pixels (bilinear interpolation)
- Center cropping to 224x224 pixels
- Rescaling to the [0, 1] range
- Normalizing with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`

The Mobilint compilation API performs these preprocessing steps internally and fuses operations like normalization directly into the MXQ model to maximize NPU efficiency.

In `model_compile.py`, we define the preprocessing pipeline as follows. This pipeline is used in calibration and will fuse the normalization module into the deep learning model.

```python
preprocess_pipeline = [
    {"op": "resize", "height": 256, "width": 256, "mode": "bilinear"},
    {"op": "centerCrop", "height": 224, "width": 224},
    {
        "op": "normalize",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "scaleToUint8": True,  # [0, 255] -> [0, 1]
        "fuseIntoFirstLayer": True, # fuse into MXQ
    },
]  # preprocessing operations for resnet 50

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
    quantization_output=0,  # 0 for layer, 1 for channel
    quantization_mode=1,  # maxpercentile
    percentile=0.9999,  # quantization percentile
    topk_ratio=0.01,  # quantization topk
)
```

After configuring the settings, the code can be executed as follows.

```bash
python model_compile.py --onnx-path {path_to_onnx_model} --calib-data-path {path_to_calibration_dataset} --save-path {path_to_save_model}
```

**What it does:**

- Loads the ONNX model
- Loads the calibration data
- Compiles the model to `.mxq` format

**Parameters:**

- `--onnx-path`: Path to the ONNX model
- `--calib-data-path`: Path to the calibration data
- `--save-path`: Path to save the MXQ model

**Output:**

- `{path_to_save_model}` file path containing the compiled model

For example, the command is as follows:

```bash
python model_compile.py --onnx-path ./resnet50.onnx --calib-data-path ./imagenet-1k-selected --save-path ./resnet50.mxq 
```

After executing the above command, the compiled model will be saved as `resnet50.mxq` in the current directory.
