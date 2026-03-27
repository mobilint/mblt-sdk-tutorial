import torch
from torchvision.models import ResNet50_Weights, resnet50

# Using pretrained weights:
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()
# Create a dummy input depending on the model's input shape
dummy_input = torch.randn(1, 3, 224, 224)
# Export to ONNX
torch.onnx.export(model, dummy_input, "resnet50.onnx")
