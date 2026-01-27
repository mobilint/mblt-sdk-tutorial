# Image Classification Model Inference

This tutorial provides step-by-step instructions for running inference with compiled image classification models using the Mobilint qbruntime.

This guide is a continuation of [mblt-sdk-tutorial/compilation/vision/image_classification/README.md](file:///workspace/mblt-sdk-tutorial/compilation/vision/image_classification/README.md). It is assumed that you have successfully compiled the model and have the following file ready:

- `./resnet50.mxq` - Compiled model file

## Prerequisites

Before running inference, ensure you have the following components installed and available:

- `qbruntime` library (to access the NPU accelerator)
- Compiled `.mxq` model file
- Python packages: `PIL`, `numpy`, `torch`

## Overview

The inference logic is implemented in the `inference_mxq.py` script. This script demonstrates the following workflow:

1.  **Model Loading**: Load the compiled `.mxq` model using `qbruntime`.
2.  **Preprocessing**: Prepare the input image (resize, center crop).
3.  **Inference**: Execute the model on the NPU accelerator.
4.  **Result Display**: Print the top-5 classification results with their associated probabilities.

## Running Inference

The `inference_mxq.py` script performs inference in several detailed steps.

First, initialize the NPU accelerator and the model configuration.

```python
acc = qbruntime.Accelerator(0)
mc = qbruntime.ModelConfig()
mc.set_single_core_mode(1)
mxq_model = qbruntime.Model(args.mxq_path, mc)
mxq_model.launch(acc)
```

Next, load and preprocess the input image. Since the normalization operation is fused into the MXQ model during compilation, the input image should remain in `UInt8` format.

```python
def preprocess_resnet50(img_path: str) -> np.ndarray:
    """Preprocess the image for ResNet-50"""
    img = Image.open(img_path).convert("RGB")
    resize_size = 256
    crop_size = (224, 224)
    out = F.pil_to_tensor(img)
    out = F.resize(out, size=resize_size, interpolation=InterpolationMode.BILINEAR)
    out = F.center_crop(out, output_size=crop_size)
    out = np.transpose(out.numpy(), axes=[1, 2, 0])
    out = out.astype(np.uint8)
    return out

image = preprocess_resnet50(args.image_path)
```

Finally, run inference and obtain the prediction probabilities.

```python
output = mxq_model.infer(image)

output = output[0].reshape(-1).astype(np.float32)
output = np.exp(output) / np.sum(np.exp(output)) # softmax
```

To run the example inference script, use the following command:

```bash
python inference_mxq.py --mxq_path ../../../compilation/vision/image_classification/resnet50.mxq --image_path ../rc/volcano.jpg
```

### Script Breakdown

- **Model Execution**: Loads the `.mxq` file onto the NPU.
- **Preprocessing**: Resizes the image to 256px, performs a center crop to 224x224, and keeps data in `UInt8` format (normalization is done on the NPU).
- **Inference**: Runs the forward pass on the NPU.
- **Result Display**: Outputs the top-5 predicted classes along with their confidence scores.

### Expected Output

The script will display the image shape and the top-5 predicted classes with their confidence scores.
