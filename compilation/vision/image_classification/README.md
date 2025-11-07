# Image Classification

This tutorial provides detailed instructions for compiling image classification models using the Mobilint qb compiler.

In this tutorial, we will use the [ResNet-50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) model, which is pretrained on the ImageNet dataset by PyTorch.

## Model Preparation

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

Calibration dataset