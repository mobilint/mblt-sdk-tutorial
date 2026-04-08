# Face Detection Model Compilation

This tutorial explains how to compile a face detection model with Mobilint `qbcompiler`.

The overall flow is intentionally similar to [../object_detection/README.md](../object_detection/README.md):

1. Prepare a pretrained model and export it to ONNX.
2. Build a representative calibration dataset.
3. Compile the model to Mobilint `.mxq` format.

For this example, we use the [YOLOv12m-face](https://github.com/akanametov/yolo-face) model from the `yolo-face` project. It is a single-class detector trained for face bounding boxes and uses `640x640` letterbox preprocessing.

## Prerequisites

Before starting, make sure the following are available:

- `qbcompiler`
- Python packages: `ultralytics`, `huggingface_hub`

Install the required Python packages with:

```bash
pip install ultralytics huggingface_hub
```

If Hugging Face access is required in your environment, authenticate before downloading the calibration dataset:

```bash
hf auth login --token <your_huggingface_token>
```

## Overview

The face detection compilation workflow has three stages:

1. **Model Preparation**: Download the pretrained face detector and export it to ONNX.
2. **Calibration Dataset Preparation**: Create a small but representative calibration set from WIDER FACE.
3. **Model Compilation**: Compile the ONNX model to `.mxq` using the selected images.

## Step 1: Model Preparation

Use `prepare_model.py` to download the pretrained YOLO face weights and export them to ONNX.

```bash
python prepare_model.py
```

**What this does:**

- Downloads `yolov12m-face.pt` from the upstream release if it is not already present.
- Loads the weights with `ultralytics.YOLO`.
- Exports the model to `yolov12m-face.onnx`.

**Output:**

- `yolov12m-face.pt`
- `yolov12m-face.onnx`

## Step 2: Calibration Dataset Preparation

As in the object detection tutorial, calibration data should match the image distribution expected during deployment. For face detection, this tutorial uses the [WIDER FACE](https://huggingface.co/datasets/CUHK-CSE/wider_face) training archive hosted on Hugging Face.

Run the dataset preparation script:

```bash
python prepare_widerface.py
```

The script downloads `WIDER_train.zip`, groups training images by sub-category, selects one random image per sub-category, and copies those images into `widerface-selected/`.

You can also choose a custom output directory or random seed:

```bash
python prepare_widerface.py --output-dir ./widerface-selected --seed 42
```

**What this does:**

- Downloads `WIDER_train.zip` from Hugging Face.
- Reads images under `WIDER_train/images`.
- Groups images by WIDER FACE sub-category.
- Randomly selects one image from each sub-category.
- Saves the selected images into `widerface-selected/`.

**Output:**

- `widerface-selected/`: Calibration dataset used during compilation

## Step 3: Model Compilation

Before compiling, confirm the preprocessing required by the model. Like the object detection YOLO example, this tutorial uses letterbox resizing so the aspect ratio is preserved while fitting the `640x640` model input.

In `model_compile.py`, the preprocessing pipeline is defined as follows:

```python
preprocess_pipeline = [{"op": "letterbox", "height": 640, "width": 640, "padValue": 114}]

preprocessing_config = PreprocessingConfig(
    apply=True,
    auto_convert_format=True,
    pipeline=preprocess_pipeline,
    input_configs={},
)
```

The script also enables `Uint8` input handling and uses the following calibration settings:

```python
calibration_config = CalibrationConfig(
    method=1,  # 0 for per tensor, 1 for per channel
    output=1,  # 0 for layer, 1 for channel
    mode=1,  # maxpercentile
    max_percentile={
        "percentile": 0.9999,
        "topk_ratio": 0.01,
    },
)
```

Run the compiler with the ONNX model and calibration dataset:

```bash
python model_compile.py --onnx-path ./yolov12m-face.onnx --calib-data-path ./widerface-selected --save-path ./yolov12m-face.mxq
```

This example passes the paths explicitly so the command matches the output from `prepare_model.py`.

**What this does:**

- Loads the ONNX model.
- Loads the calibration images.
- Compiles the model to `.mxq` format.
- Saves an intermediate `.mblt` graph alongside the ONNX file.

**Parameters:**

- `--onnx-path`: Path to the ONNX model file
- `--calib-data-path`: Path to the calibration image directory
- `--save-path`: Path where the compiled `.mxq` model will be written

**Output:**

- `yolov12m-face.mxq`
- `yolov12m-face.mblt`

After the command finishes, continue to [../../runtime/face_detection/README.md](../../runtime/face_detection/README.md) to run inference with the compiled model.
