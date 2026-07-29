# Semantic Segmentation Model Compilation

This tutorial explains how to compile the Ultralytics `YOLO26m-sem` Cityscapes semantic-segmentation model into Mobilint `.mxq` and `.mblt` artifacts with `qbcompiler`.

The model accepts RGB input and uses YOLO-style letterbox preprocessing at `1024x2048`. Calibration images are selected from the `image` column of the `validation` split in [`Chris1/cityscapes_segmentation`](https://huggingface.co/datasets/Chris1/cityscapes_segmentation).

## Prerequisites

Before starting, install `qbcompiler` and the Python packages used by this tutorial:

```bash
pip install ultralytics datasets opencv-python
```

The public dataset does not require authentication. Its validation split contains 500 `1024x2048` image and semantic-mask pairs and occupies approximately 1.2 GB in Parquet format. Hugging Face `datasets` caches downloaded files, so make sure sufficient disk space is available.

## Overview

1. Export `YOLO26m-sem` to ONNX.
2. Select calibration images from the Cityscapes validation split.
3. Compile the ONNX model with `1024x2048` letterbox preprocessing.

## Step 1: Prepare the Model

Export the pretrained model to ONNX:

```bash
yolo export model=yolo26m-sem.pt format=onnx imgsz=1024,2048
```

The command writes `yolo26m-sem.onnx` to the current directory. If the supplied `yolo26m-sem.onnx` is already in this directory, you can use it directly.

The supplied ONNX model has these input and output shapes:

- Input: `[1, 3, 1024, 2048]`
- Output: `[1, 1024, 2048]`

## Step 2: Prepare the Calibration Dataset

`prepare_cityscapes.py` queries the Hugging Face Dataset Viewer for the dataset's Parquet files and passes only the `validation` shard URLs to `load_dataset`. The `train` and `test` Parquet files are not downloaded. The loader also reads only the `image` column, and semantic masks are not saved because compilation calibrates the RGB model input only.

The validation split consists of two Parquet files totaling approximately 1.2 GB. By default, the script uses seed `42` to select 100 of the 500 validation images and saves them to `./cityscapes-selected`:

```bash
python prepare_cityscapes.py
```

To change the number of images, output directory, or seed:

```bash
python prepare_cityscapes.py \
  --num-images 200 \
  --output-dir ./cityscapes-calibration \
  --seed 7
```

If the output directory already exists, add `--overwrite` to replace it.

## Step 2-1 (Optional): Convert Images to Preprocessed Tensors

`qbcompiler` can usually consume the raw image directory and apply preprocessing during calibration. Use this optional step when you need explicit, reusable calibration tensors.

`convert_img_to_tensor.py` applies:

- BGR-to-RGB conversion
- Aspect-ratio-preserving letterbox resize to `1024x2048`
- Constant padding with value `114`
- `/255` conversion to `float32` HWC tensors

```bash
python convert_img_to_tensor.py
```

The script writes tensors to `calib_data_tensor`. To compile from these tensors, use that directory as `calib_data_path` and omit the built-in preprocessing configuration. The provided `model_compile.py` intentionally uses raw images.

## Step 3: Compile the Model

`model_compile.py` applies this preprocessing configuration to the raw RGB calibration images:

```python
preprocessing_config = PreprocessingConfig(
    apply=True,
    auto_convert_format=True,
    pipeline=[
        {
            "op": "letterbox",
            "height": 1024,
            "width": 2048,
            "padValue": 114,
        }
    ],
    input_configs={},
)
```

The `/255` normalization is fused into the MXQ model with `Uint8InputConfig`, so the compiled model accepts `uint8` input. Letterboxing is a spatial operation and must still be applied before runtime inference.

`model_compile.py` automatically uses CUDA for MXQ compilation when `torch.cuda.is_available()` is true. In a CPU-only `qbcompiler` image, it selects CPU compilation instead. The selected host device is printed before compilation starts.

Compile for the target NPU:

```bash
# ARIES
python model_compile.py --target-device aries-rb

# REGULUS
python model_compile.py --target-device regulus-rb
```

The command generates:

- `yolo26m-sem.mxq`: quantized NPU model
- `yolo26m-sem.mblt`: intermediate graph for inspection

All paths can be customized:

```bash
python model_compile.py \
  --onnx-path ./yolo26m-sem.onnx \
  --calib-data-path ./cityscapes-selected \
  --save-path ./yolo26m-sem.mxq \
  --mblt-path ./yolo26m-sem.mblt \
  --target-device aries-rb
```
