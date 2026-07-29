# Pose Estimation Model Compilation

This tutorial explains how to compile a pose estimation model with Mobilint `qbcompiler`.

The example uses [YOLO11m-pose](https://docs.ultralytics.com/models/yolo11/), pretrained by Ultralytics on the COCO dataset. The model estimates skeletal keypoints for objects in an image.

## Prerequisites

Before you begin, make sure the following are available:

- `qbcompiler`
- A Hugging Face account with access to the gated COCO dataset

Install the required Python packages:

```bash
pip install ultralytics aiohttp aiofiles
```

## Overview

The compilation workflow has three main steps:

1. **Prepare the model**: Download the model and export it to ONNX.
2. **Prepare the calibration dataset**: Build a representative calibration dataset from COCO.
3. **Compile the model**: Convert the ONNX model to `.mxq` using the calibration data.

## Step 1: Prepare the Model

Use the `ultralytics` package to download the pretrained model and export it to ONNX:

```bash
yolo export model=yolo11m-pose.pt format=onnx
```

After the command finishes, the exported model is saved as `yolo11m-pose.onnx` in the current directory.

## Step 2: Prepare the Calibration Dataset

The calibration dataset should reflect the model's typical input distribution. Because YOLO11m-pose was trained on the [COCO dataset](https://cocodataset.org/#download), this tutorial uses COCO samples for calibration.

Before accessing the dataset, sign in to [Hugging Face](https://huggingface.co/) and authenticate with your token:

```bash
hf auth login --token <your_huggingface_token>
```

If you do not know your token, check your [Hugging Face account settings](https://huggingface.co/settings/tokens).

Run `prepare_coco.py` to automate dataset preparation. The script reads COCO image URLs, randomly selects samples, and downloads them into the `coco-selected` directory.

```bash
python prepare_coco.py
```

**What this script does:**

- Downloads COCO image URLs from Hugging Face
- Randomly selects images for the calibration dataset
- Saves the selected images in `coco-selected`

**Output:**

- `coco-selected`: Calibration dataset directory

The downloaded images in `coco-selected` will be used as the calibration dataset.

## Step 2-1 (Optional): Convert Images to Preprocessed Tensors

You can also prepare the calibration dataset as preprocessed `.npy` tensors. This is useful when your model requires custom preprocessing and you want to generate calibration inputs yourself.

Since `qbcompiler` v1.0.0, the standard image-based calibration flow is usually sufficient. Use this optional step only when you need explicit control over preprocessing.

The conversion script assumes a preprocessing function that:

- Accepts an image path as input
- Returns a NumPy tensor
- Produces tensors in `HWC` format for calibration

Example preprocessing function:

```python
def pre_ftn(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h0, w0 = img.shape[:2]
    r = min(640 / h0, 640 / w0)
    new_unpad = int(round(w0 * r)), int(round(h0 * r))
    dh, dw = 640 - new_unpad[1], 640 - new_unpad[0]

    dw /= 2
    dh /= 2
    if (img.shape[1], img.shape[0]) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
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
    remove_npy=True,
)
```

Run the script:

```bash
python convert_img_to_tensor.py
```

By default, it reads images from `./coco-selected` and writes the tensor dataset to `./calib_data_tensor`.

## Step 3: Compile the Model

Before compiling, confirm the preprocessing requirements. YOLO models typically use the `LetterBox` operation, as described in the [Ultralytics repository](https://github.com/ultralytics/ultralytics).

The Mobilint compilation API applies the preprocessing pipeline during calibration. The normalization step (`/255` scaling) is fused into the MXQ model through `Uint8InputConfig`, which lets the runtime model accept `uint8` input directly. Spatial transforms such as letterboxing are not fused, so they still need to be applied at runtime.

In `model_compile.py`, the preprocessing pipeline is defined as follows:

```python
preprocess_pipeline = [
    {
        "op": "letterbox",
        "height": 640,
        "width": 640,
        "padValue": 114,
    }
]
preprocessing_config = PreprocessingConfig(
    apply=True,
    auto_convert_format=True,
    pipeline=preprocess_pipeline,
    input_configs={},
)
```

When preprocessing fusion is enabled, set the MXQ input type to `uint8`:

```python
mxq_compile(
    # ... model, calibration data, backend, and target device settings
    preprocessing_config=preprocessing_config,
    uint8_input_config=Uint8InputConfig(apply=True, inputs=[]),
    calibration_config=calibration_config,
)
```

If you want to keep the original input format, disable both preprocessing fusion and `Uint8InputConfig`.

The example uses the following quantization configuration:

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

After configuring the settings, run `model_compile.py` with `--target-device` for your hardware. A single run produces both outputs: the quantized MXQ file (`--save-path`) and the intermediate MBLT graph (`--mblt-path`).

For MXQ compilation, `model_compile.py` automatically uses CUDA when `torch.cuda.is_available()` is true and otherwise falls back to CPU. This supports both GPU-enabled and CPU-only `qbcompiler` images, and the selected host device is printed before compilation starts.

## Step 3-1 (Optional): Compile with Prepared Tensor Files

If you already prepared `.npy` tensor files, you can use that directory as `calib_data_path` instead of providing raw images and a preprocessing pipeline.

```python
mxq_compile(
    model=args.onnx_path,
    calib_data_path=args.calib_data_path,
    save_path=args.save_path,
    image_channels=3,
    backend="onnx",
    device="gpu",
    target_device=args.target_device,
    inference_scheme=inferece_sheme,
    calibration_config=calibration_config,
)
```

**Parameters:**

- `--onnx-path`: Path to the ONNX model
- `--calib-data-path`: Path to the calibration data
- `--save-path`: Path to save the MXQ model (`onnx -> mxq`)
- `--mblt-path`: Path to save the MBLT intermediate graph (`onnx -> mblt`)
- `--target-device` (required): Target NPU. See the table below. The inference scheme is selected automatically (`ARIES = all`, `REGULUS = single`).

**Output:**

- MXQ model at `--save-path` (`onnx -> mxq`, quantized NPU package)
- MBLT intermediate graph at `--mblt-path` (`onnx -> mblt`, pre-quantization graph)

### Select the Target Device (`--target-device`)

The required model depends on the target device. Older REGULUS hardware (`regulus-ra`, customers before 2026-06) supports YOLOv9 and earlier, so it uses `yolov8m-pose`. ARIES (`aries-rb`) and newer REGULUS hardware (`regulus-rb`, customers from 2026-06) use `yolo11m-pose`.

| User | `--target-device` | Model |
| --- | --- | --- |
| ARIES | `aries-rb` | `yolo11m-pose` |
| REGULUS (customers before 2026-06) | `regulus-ra` | `yolov8m-pose` |
| REGULUS (customers from 2026-06) | `regulus-rb` | `yolo11m-pose` |

Export the model that matches your target device first. Step 1 uses `yolo11m-pose`; for `regulus-ra`, export `yolov8m-pose` instead.

```bash
# YOLO11 pose (for aries-rb / regulus-rb)
yolo export model=yolo11m-pose.pt format=onnx

# YOLOv8 pose (for regulus-ra)
yolo export model=yolov8m-pose.pt format=onnx
```

```bash
# ARIES
python model_compile.py --onnx-path ./yolo11m-pose.onnx --calib-data-path ./coco-selected --save-path ./yolo11m-pose.mxq --mblt-path ./yolo11m-pose.mblt --target-device aries-rb

# REGULUS (customers before 2026-06)
python model_compile.py --onnx-path ./yolov8m-pose.onnx --calib-data-path ./coco-selected --save-path ./yolov8m-pose.mxq --mblt-path ./yolov8m-pose.mblt --target-device regulus-ra

# REGULUS (customers from 2026-06)
python model_compile.py --onnx-path ./yolo11m-pose.onnx --calib-data-path ./coco-selected --save-path ./yolo11m-pose.mxq --mblt-path ./yolo11m-pose.mblt --target-device regulus-rb
```
