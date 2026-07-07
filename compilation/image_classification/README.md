# Image Classification Model Compilation

This tutorial explains how to compile an image classification model with Mobilint `qbcompiler`.

The example uses [ResNet-50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) from `torchvision`. This model is pretrained on ImageNet-1K and is a standard benchmark for classifying images into 1,000 categories.

## Prerequisites

Before you begin, make sure the following are available:

- `qbcompiler`
- A Hugging Face account with access to the gated ImageNet dataset

Install the package used to download the dataset:

```bash
pip install datasets
```text

## Overview

The workflow has three main steps:

1. **Prepare the model**: Download ResNet-50 and export it to ONNX.
2. **Prepare the calibration dataset**: Build a representative calibration set from ImageNet.
3. **Compile the model**: Convert the ONNX model to `.mxq` using the calibration data.

## Step 1: Prepare the Model

Use `torchvision` to download the pretrained model, then export it to ONNX with `torch.onnx.export()`.

```python
import torch
from torchvision.models import ResNet50_Weights, resnet50

# Use pretrained weights.
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# Create a dummy input that matches the model input shape.
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX.
torch.onnx.export(model, (dummy_input,), "resnet50.onnx")
```text

Run the script:

```bash
python prepare_model.py
```text

This saves the exported model as `resnet50.onnx` in the current directory.

## Step 2: Prepare the Calibration Dataset Source

The calibration dataset is used to collect activation statistics for quantization. For this ResNet-50 example, use the [ImageNet dataset](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

Before downloading the dataset:

- Create a [Hugging Face](https://huggingface.co/) account.
- Accept the dataset terms on the [ImageNet-1K dataset page](https://huggingface.co/datasets/ILSVRC/imagenet-1k).

Then log in with your Hugging Face token:

```bash
hf auth login --token <your_huggingface_token>
```text

If you do not know your token, check your [Hugging Face token settings](https://huggingface.co/settings/tokens).

Next, run the dataset preparation script:

```bash
python prepare_imagenet.py
```text

What this script does:

- Downloads the validation split from Hugging Face
- Selects one image for each of the 1,000 classes
- Saves the selected images to `imagenet-1k-selected/`

**Output:**

- `imagenet-1k-selected/`, containing 1,000 images

This directory is the calibration image set used in the next step.

## Step 2-1 (Optional): Convert Images to Preprocessed Tensors

The calibration data can also be prepared as preprocessed `.npy` tensors. This is useful when your model requires a custom preprocessing function and you want to generate the tensor inputs yourself.

Since `qbcompiler` v1.0.0, a standardized calibration dataset generation flow is available, so you can usually skip this step. Use it only when you need explicit control over preprocessing.

The conversion script assumes a preprocessing function that:

- Takes an image path as input
- Returns a NumPy tensor
- Produces tensors in `HWC` format for calibration data

Example preprocessing function:

```python
def pre_ftn(img_path):
    img = Image.open(img_path).convert("RGB")
    preprocess_pipeline = [
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop((224, 224)),
        T.ToTensor(),  # [0, 255] -> [0, 1]
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    preprocess = T.Compose(preprocess_pipeline)
    tensor = cast(torch.Tensor, preprocess(img))
    return tensor.permute((1, 2, 0)).numpy()  # (C, H, W) -> (H, W, C)
```text

The script uses `make_calib_man()` to generate the tensor dataset:

```python
make_calib_man(
    pre_ftn=pre_ftn,
    data_dir=args.source_path,
    save_dir=os.path.dirname(args.npy_path),
    save_name=os.path.basename(args.npy_path),
    remove_npy=True,  # Clean the destination before writing new .npy files.
)
```text

Run the script:

```bash
python convert_img_to_tensor.py
```text

By default, it reads images from `./imagenet-1k-selected` and writes the tensor dataset under `./calib_data_tensor`.

## Step 3: Compile the Model

Before compilation, confirm the preprocessing required by the model. According to the [official ResNet-50 documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html), the expected preprocessing is:

- Resize the shorter side to 256 pixels with bilinear interpolation
- Center crop to `224x224`
- Rescale pixel values to `[0, 1]`
- Normalize with mean `[0.485, 0.456, 0.406]`
- Normalize with standard deviation `[0.229, 0.224, 0.225]`

For this tutorial, `qbcompiler` applies that preprocessing through a standardized preprocessing pipeline:

```python
preprocess_pipeline = [
    {"op": "resize", "height": 256, "width": 256, "mode": "bilinear"},
    {"op": "centerCrop", "height": 224, "width": 224},
    {
        "op": "normalize",
        "scaleToUint8": True,  # [0, 255] -> [0, 1]
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "fuseIntoFirstLayer": True,
    },
]  # Preprocessing operations for ResNet-50.

preprocessing_config = PreprocessingConfig(
    apply=True,
    auto_convert_format=True,
    pipeline=preprocess_pipeline,
    input_configs={},
)
```text

The normalization step, including `/255` scaling, is fused into the MXQ model through `fuseIntoFirstLayer` and `Uint8InputConfig`. This lets the compiled model accept `uint8` input directly at runtime. Spatial transforms such as `resize` and `centerCrop` are not fused, so they still need to be applied at runtime.

When you enable preprocessing fusion, set the MXQ input type to `uint8`:

```python
# ONNX -> MXQ: quantized package that runs on the NPU
mxq_compile(
    # ... model, calibration data, backend, and target device settings
    preprocessing_config=preprocessing_config,
    uint8_input_config=Uint8InputConfig(apply=True, inputs=[]),
    calibration_config=calibration_config,
)
```text

If you want to keep the original input format, disable both `fuseIntoFirstLayer` and `Uint8InputConfig`.

The example also uses the following quantization configuration:

```python
calibration_config = CalibrationConfig(
    method=1,  # 0 for per tensor, 1 for per channel
    output=0,  # 0 for layer, 1 for channel
    mode=1,  # maxpercentile
    max_percentile={
        "percentile": 0.9999,  # quantization percentile
        "topk_ratio": 0.01,  # quantization top-k ratio
    },
)
```text

After the settings are configured, run `model_compile.py` with `--target-device`. The same script supports both ARIES and REGULUS devices.

## Step 3-1 (Optional): Compile with Prepared Tensor Files

If you already prepared `.npy` tensor files, you can use that directory as `calib_data_path` instead of providing raw image files and a preprocessing pipeline.

```python
mxq_compile(
    model=args.onnx_path,
    calib_data_path=args.calib_data_path,  # Directory of .npy files, or a .txt file listing them
    save_path=args.save_path,
    image_channels=3,  # Convert grayscale calibration images to RGB if needed
    backend="onnx",
    device="gpu",
    target_device=args.target_device,
    inference_scheme=inferece_sheme,
    calibration_config=calibration_config,
)
```text

Parameters:

- `--onnx-path`: Path to the ONNX model
- `--calib-data-path`: Path to the calibration data
- `--save-path`: Path to save the MXQ model (`onnx -> mxq`)
- `--mblt-path`: Path to save the MBLT intermediate graph (`onnx -> mblt`)
- `--target-device` (required): Target NPU. See the table below. The inference scheme is selected automatically (`ARIES = all`, `REGULUS = single`).

**Output:**

- MXQ model at `--save-path` (`onnx -> mxq`, quantized NPU package)
- MBLT graph at `--mblt-path` (`onnx -> mblt`, pre-quantization intermediate graph)

### Select the Target Device (`--target-device`)

| User | `--target-device` |
| --- | --- |
| ARIES | `aries-rb` |
| REGULUS (customers before 2026-06) | `regulus-ra` |
| REGULUS (customers from 2026-06) | `regulus-rb` |

```bash
# ARIES
python model_compile.py --onnx-path ./resnet50.onnx --calib-data-path ./imagenet-1k-selected --save-path ./resnet50.mxq --mblt-path ./resnet50.mblt --target-device aries-rb

# REGULUS (customers before 2026-06)
python model_compile.py --onnx-path ./resnet50.onnx --calib-data-path ./imagenet-1k-selected --save-path ./resnet50.mxq --mblt-path ./resnet50.mblt --target-device regulus-ra

# REGULUS (customers from 2026-06)
python model_compile.py --onnx-path ./resnet50.onnx --calib-data-path ./imagenet-1k-selected --save-path ./resnet50.mxq --mblt-path ./resnet50.mblt --target-device regulus-rb
```text

After the command finishes, `resnet50.mxq` and `resnet50.mblt` are saved in the current directory.
