# Image Classification Model Inference

This tutorial provides detailed instructions for running inference with compiled image classification models using the Mobilint qb runtime.

This guide continues from `mblt-sdk-tutorial/compilation/vision/image_classification/README.md`. We assume you have successfully compiled the model and have the following files ready:

- `./resnet50.mxq` - Compiled model file

## Prerequisites

Before running inference, ensure you have:

- maccel runtime library (provides NPU accelerator access)
- Compiled `.mxq` model file
- Python packages: `PIL`, `numpy`, `torch`, `torchvision`

## Overview

The inference process is implemented in the `inference_mxq.py` script. This script demonstrates how to:

- Load the compiled `.mxq` model using maccel runtime
- Preprocess the input image (resize, crop, normalize)
- Run inference on the NPU accelerator
- Print the top-5 classification results with probabilities

## Running Inference

To run the example inference script:

```bash
python inference_mxq.py --mxq_path ../../../compilation/vision/image_classification/resnet50.mxq --image_path ../rc/volcano.jpg
```

**What this does:**

- Loads the compiled `.mxq` model onto the NPU accelerator
- Loads and preprocesses the sample image (ResNet-50 preprocessing: resize to 256px, center crop to 224x224, normalize)
- Runs inference on the NPU accelerator
- Prints the top-5 classification results with their probabilities

**Expected output:**

The script will display the image shape and the top-5 predicted classes with their confidence scores.
