# Depth Estimation Model Compilation

This tutorial explains how to compile the Ultralytics `YOLO26m-depth` model into Mobilint `.mxq` and `.mblt` artifacts with `qbcompiler`.

The model accepts RGB input and uses YOLO-style letterbox preprocessing at `768x768`. This tutorial selects calibration images from the NYU Depth V2 validation set in the [Ultralytics dataset archive](https://github.com/ultralytics/assets/releases/download/v0.0.0/nyu-depth.zip).

## Prerequisites

Before you begin, install `qbcompiler` and the Python packages used by this tutorial:

```bash
pip install ultralytics opencv-python
```

The compressed dataset archive is about 502 MB and extracts to about 1.5 GB.

## Overview

1. Export `YOLO26m-depth` to ONNX.
2. Download NYU Depth V2 and select validation images for calibration.
3. Compile the ONNX model with 768x768 letterbox preprocessing.

## Step 1: Prepare the Model

Export the pretrained model to ONNX:

```bash
yolo export model=yolo26m-depth.pt format=onnx
```

The command writes `yolo26m-depth.onnx` to the current directory. If the supplied `yolo26m-depth.onnx` is already in this directory, you can use it directly.

## Step 2: Prepare the Calibration Dataset

`prepare_nyu_depth_v2.py` downloads and extracts the [Ultralytics NYU Depth V2 archive](https://github.com/ultralytics/assets/releases/download/v0.0.0/nyu-depth.zip). It then makes a reproducible selection of RGB images from `nyu-depth/images/val` for calibration. Depth maps are not copied because the compiler only uses RGB model inputs.

By default, the script downloads and extracts the archive in a temporary directory. It then uses seed `42` to select 100 validation images and copies them to `./nyu-depth-selected`. The temporary archive and extracted dataset are removed automatically.

```bash
python prepare_nyu_depth_v2.py
```

To select a different number of images or use a different output directory:

```bash
python prepare_nyu_depth_v2.py --num-images 200 --output-dir ./nyu-depth-calibration --seed 7
```

If the output directory already exists, add `--overwrite` to replace it.

## Step 2-1 (Optional): Convert Images to Preprocessed Tensors

`qbcompiler` can usually consume the raw image directory and apply the preprocessing configuration during calibration. Use this optional step when you need explicit, reusable calibration tensors.

`convert_img_to_tensor.py` applies the same preprocessing as the compiler path:

- BGR to RGB conversion
- Aspect-ratio-preserving letterbox resize to `768x768`
- Constant padding with value `114`
- `/255` conversion to `float32` HWC tensors

```bash
python convert_img_to_tensor.py
```

The script writes the tensors to `calib_data_tensor`. To use them with `mxq_compile`, replace the raw-image calibration path and omit the built-in preprocessing configuration. The provided `model_compile.py` intentionally uses the raw-image workflow.

## Step 3: Compile the Model

`model_compile.py` applies this preprocessing configuration to the raw RGB calibration images:

```python
preprocessing_config = PreprocessingConfig(
    apply=True,
    auto_convert_format=True,
    pipeline=[
        {"op": "letterbox", "height": 768, "width": 768, "padValue": 114},
        {
            "op": "normalize",
            "scaleToUint8": True,
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "fuseIntoFirstLayer": True,
        },
    ],
    input_configs={},
)
```

The `/255` normalization is fused into the MXQ model with `Uint8InputConfig`, so the compiled model accepts `uint8` input. Letterboxing is a spatial operation and must still be applied before inference.

`model_compile.py` automatically uses CUDA for MXQ compilation when `torch.cuda.is_available()` is true. In a CPU-only `qbcompiler` image, it selects CPU compilation instead. The selected host device is printed before compilation starts.

Compile for your target NPU:

```bash
# ARIES
python model_compile.py --target-device aries-rb

# REGULUS
python model_compile.py --target-device regulus-rb
```

> **Note:** The YOLO26 depth model is not available for older REGULUS devices that use `regulus-ra`. Use `aries-rb` or `regulus-rb`.

The command generates:

- `yolo26m-depth.mxq`: quantized NPU model
- `yolo26m-depth.mblt`: intermediate graph for inspection

All paths can be customized:

```bash
python model_compile.py \
  --onnx-path ./yolo26m-depth.onnx \
  --calib-data-path ./nyu-depth-selected \
  --save-path ./yolo26m-depth.mxq \
  --mblt-path ./yolo26m-depth.mblt \
  --target-device aries-rb
```
