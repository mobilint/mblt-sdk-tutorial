# Image Classification Runtime

This tutorial explains how to run a compiled image classification MXQ model with Mobilint `qbruntime`.

Before starting, complete the compilation flow in [../../../compilation/image_classification/README.md](../../../compilation/image_classification/README.md). The runtime example in this directory expects the compiled model at `../../../compilation/image_classification/resnet50.mxq`.

## Prerequisites

Make sure the following components are available:

- Mobilint `qbruntime`
- A compiled `.mxq` model file
- Python packages: `Pillow`, `numpy`, `torch`, `torchvision`

If the Python packages are not already installed in your environment, install them with:

```bash
pip install pillow numpy torch torchvision
```

## Overview

The runtime flow is implemented in `inference_mxq.py` and follows these steps:

1. Load the compiled ResNet-50 MXQ model with `qbruntime`.
2. Read the input image and apply resize plus center-crop preprocessing.
3. Run inference on the Mobilint NPU.
4. Convert logits to probabilities with softmax.
5. Print the top-5 ImageNet predictions.

The compiled MXQ model typically includes normalization, so this example keeps the runtime input in `uint8` format.

## Files in This Tutorial

- `inference_mxq.py`: Runs the full inference flow and prints the top-5 predictions.
- `imagenet.py`: Maps class indices to ImageNet labels.

## How the Script Works

The script first initializes the accelerator and launches the compiled model:

```python
acc = qbruntime.Accelerator(0)
mc = qbruntime.ModelConfig()
mc.set_single_core_mode(None, [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)])
mxq_model = qbruntime.Model(args.mxq_path, mc)
mxq_model.launch(acc)
```

Next, it reads the image, resizes it to `256`, applies a `224x224` center crop, and converts the result into an HWC NumPy array:

```python
def preprocess_resnet50(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    resize_size = [256]
    crop_size = [224, 224]
    out = F.pil_to_tensor(img)
    out = F.resize(out, size=resize_size, interpolation=InterpolationMode.BILINEAR)
    out = F.center_crop(out, output_size=crop_size)
    out = np.transpose(out.numpy(), axes=[1, 2, 0])
    # Option 1: normalization is fused into the model/runtime.
    out = out.astype(np.uint8)

    # Option 2: normalization is not fused.
    # out = out.astype(np.float32) / 255.0
    # out = (out - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
    #       np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return out
```

After inference, the script reshapes the output logits, applies softmax, and prints the top-5 ImageNet classes with their probabilities.

## Run the Example

Run the tutorial with the default sample paths:

```bash
python inference_mxq.py
```

This command uses the following defaults:

- Model: `../../../compilation/image_classification/resnet50.mxq`
- Input image: `../rc/volcano.jpg`

To pass the paths explicitly, run:

```bash
python inference_mxq.py --mxq-path ../../../compilation/image_classification/resnet50.mxq --image-path ../rc/volcano.jpg
```

## Parameters

- `--mxq-path`: Path to the compiled `.mxq` model.
- `--image-path`: Path to the input image.

## Expected Output

The script prints the preprocessed image shape and the top-5 ImageNet predictions with their probabilities.
