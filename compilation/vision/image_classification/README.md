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

## Calibration Dataset Preparation

Calibration dataset is a set of images that represent the typical input distribution of the model. We will use the [ImageNet dataset](https://huggingface.co/datasets/ILSVRC/imagenet-1k) for this tutorial.

Before using the dataset, sign up for an account on [HuggingFace](https://huggingface.co/) and accept the agreement to use the dataset on the [dataset page](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

Then, log in to HuggingFace using the following command and replace <your_huggingface_token> with your actual HuggingFace token:

```bash
hf auth login --token <your_huggingface_token>
```

If you are not sure about your HuggingFace token, you can find it in your [HuggingFace account settings](https://huggingface.co/settings/tokens).



