# Face Detection Model Compilation

This tutorial explains how to compile a face detection model with Mobilint `qbcompiler`.

The overall flow is intentionally similar to [../object_detection/README.md](../object_detection/README.md):

1. Prepare a pretrained model and export it to ONNX.
2. Build a representative calibration dataset.
3. Compile the model to Mobilint `.mxq` format.

In this example, we use the [YOLOv12m-face](https://github.com/akanametov/yolo-face) model from the `yolo-face` project. It is a single-class detector trained for face bounding boxes and uses `640x640` letterbox preprocessing.

## Prerequisites

Before you begin, make sure the following are available:

- `qbcompiler`
- Python packages: `ultralytics`, `huggingface_hub`

Install the required Python packages with:

```bash
pip install ultralytics huggingface_hub
```

If your environment requires Hugging Face authentication, sign in before downloading the calibration dataset:

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

- Downloads `yolov12m-face.pt` from the upstream release if it is not already available locally.
- Loads the weights with `ultralytics.YOLO`.
- Exports the model to `yolov12m-face.onnx`.

**Output:**

- `yolov12m-face.pt`
- `yolov12m-face.onnx`

## Step 2: Calibration Dataset Preparation

As in the object detection tutorial, the calibration data should match the image distribution expected during deployment. For face detection, this tutorial uses the [WIDER FACE](https://huggingface.co/datasets/CUHK-CSE/wider_face) training archive hosted on Hugging Face.

Run the dataset preparation script:

```bash
python prepare_widerface.py
```

The script downloads `WIDER_train.zip`, groups the training images by sub-category, selects one random image per sub-category, and copies the selected images into `widerface-selected/`.

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

## Step 2-1 (Optional): Convert Images to Preprocessed Tensors

You can also prepare the calibration data as preprocessed `.npy` tensors. This is useful when your model requires a custom preprocessing function and you want to generate the tensor inputs yourself.

Since `qbcompiler` v1.0.0, a standardized calibration dataset generation flow is available, so you can usually skip this step. Use it only when you need explicit control over preprocessing.

The conversion script assumes a preprocessing function that:

- Takes an image path as input
- Returns a NumPy tensor
- Produces tensors in `HWC` format for calibration data

Example preprocessing function:

```python
def pre_ftn(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h0, w0 = img.shape[:2]  # original hw
    r = min(640 / h0, 640 / w0)  # ratio
    new_unpad = int(round(w0 * r)), int(round(h0 * r))
    dh, dw = (
        640 - new_unpad[1],
        640 - new_unpad[0],
    )  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if (img.shape[1], img.shape[0]) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # add border
    img = (img / 255).astype(np.float32)

    return img
```

The script uses `make_calib_man()` to generate the tensor dataset:

```python
make_calib_man(
    pre_ftn=pre_ftn,
    data_dir=args.source_path,
    save_dir=os.path.dirname(args.npy_path),
    save_name=os.path.basename(args.npy_path),
    remove_npy=True,  # Clean the destination before writing new .npy files.
)
```

Run the script:

```bash
python convert_img_to_tensor.py
```

By default, it reads images from `./widerface-selected` and writes the tensor dataset under `./calib_data_tensor`.

## Step 3: Model Compilation

Before compiling, confirm the preprocessing required by the model. As in the YOLO object detection example, this tutorial uses letterbox resizing so the aspect ratio is preserved while fitting the `640x640` model input.

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

As part of normalization, the `letterbox` operation includes `1/255` scaling. This preprocessing can be fused into the MXQ model through `fuseIntoFirstLayer` and `Uint8InputConfig`.

When you enable preprocessing fusion, set the MXQ input type to `uint8`:

```python
# ONNX -> MXQ: quantized package that runs on the NPU
mxq_compile(
    # ... model, calibration data, backend, and target device settings
    preprocessing_config=preprocessing_config,
    uint8_input_config=Uint8InputConfig(apply=True, inputs=[]),
    calibration_config=calibration_config,
)
```

If you want to keep the original input format, disable both `fuseIntoFirstLayer` and `Uint8InputConfig`.

The example also uses the following quantization configuration:

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
```

**Parameters:**

- `--onnx-path`: Path to the ONNX model file
- `--calib-data-path`: Path to the calibration image directory, or to the directory of prepared `.npy` files
- `--save-path`: Path to save the MXQ model (onnx -> mxq output)
- `--mblt-path`: Path to save the MBLT intermediate graph (onnx -> mblt output)
- `--target-device` (required): Target NPU. See the table below. The inference scheme is derived automatically (ARIES = `all`, REGULUS = `single`).

**Output:**

- `yolov12m-face.mxq` (onnx -> mxq, quantized NPU package)
- `yolov12m-face.mblt` (onnx -> mblt, pre-quantization graph)

### Selecting the target device (`--target-device`)

| User | `--target-device` |
| --- | --- |
| ARIES | `aries-rb` |
| REGULUS (customers from 2026-06) | `regulus-rb` |

> **Note**: Face detection uses the YOLOv12 `yolo-face` model, which is **not supported on older REGULUS (`regulus-ra`, for customers before 2026-06)**. That generation supports only YOLOv9 and earlier. Use `aries-rb` or `regulus-rb`.

```bash
# ARIES
python model_compile.py --onnx-path ./yolov12m-face.onnx --calib-data-path ./widerface-selected --save-path ./yolov12m-face.mxq --mblt-path ./yolov12m-face.mblt --target-device aries-rb

# REGULUS (customers from 2026-06)
python model_compile.py --onnx-path ./yolov12m-face.onnx --calib-data-path ./widerface-selected --save-path ./yolov12m-face.mxq --mblt-path ./yolov12m-face.mblt --target-device regulus-rb
```

After you run the command, the MXQ file (`yolov12m-face.mxq`) and MBLT file (`yolov12m-face.mblt`) are saved in the current directory.

After the command finishes, continue to [../../runtime/python/face_detection/README.md](../../runtime/python/face_detection/README.md) to run inference with the compiled model.
