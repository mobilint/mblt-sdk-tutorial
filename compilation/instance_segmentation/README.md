# Instance Segmentation Model Compilation

This tutorial explains how to compile an instance segmentation model with Mobilint `qbcompiler`.

The example uses [YOLO11m-seg](https://docs.ultralytics.com/models/yolo11/), a COCO-pretrained model from Ultralytics. The model performs instance segmentation by detecting individual objects and predicting a mask for each one.

## Prerequisites

Before you begin, make sure the following are available:

- `qbcompiler` v1.0.0
- A Hugging Face account with access to the gated COCO dataset

Install the required Python packages:

```bash
pip install ultralytics aiohttp aiofiles
```text

## Overview

The workflow has three main steps:

1. **Prepare the model**: Download the pretrained model and export it to ONNX.
2. **Prepare calibration data**: Build a representative calibration dataset from COCO.
3. **Compile the model**: Convert the ONNX model to `.mxq` using the calibration data.

## Step 1: Prepare the Model

Use the `ultralytics` CLI to export the pretrained model to ONNX:

```bash
yolo export model=yolo11m-seg.pt format=onnx
```text

After the command finishes, the exported model is saved as `yolo11m-seg.onnx` in the current directory.

## Step 2: Prepare the Calibration Dataset

The calibration dataset should reflect the model's expected input distribution. Because YOLO11m-seg is trained on the [COCO dataset](https://cocodataset.org/#download), this tutorial uses COCO samples for calibration.

Before downloading the dataset, sign in to Hugging Face with a token that has access to the dataset:

```bash
hf auth login --token <your_huggingface_token>
```text

If you do not know your token, create or copy one from your [Hugging Face token settings](https://huggingface.co/settings/tokens).

Use `prepare_coco.py` to download a random subset of COCO images into `coco-selected`:

```bash
python prepare_coco.py
```text

What the script does:

- Downloads COCO image URLs from Hugging Face
- Randomly selects images for calibration
- Saves the images in `coco-selected`

**Output:**

- `coco-selected`: calibration image directory

This directory is the calibration dataset used in the next step.

## Step 2-1 (Optional): Convert Images to Preprocessed Tensors

You can also prepare calibration inputs as preprocessed `.npy` tensors. This is useful when your model needs custom preprocessing and you want to control tensor generation explicitly.

Since `qbcompiler` v1.0.0 provides a standardized calibration data generation flow, you can usually skip this step. Use it only when you need manual preprocessing control.

The conversion script assumes a preprocessing function that:

- Takes an image path as input
- Returns a NumPy tensor
- Produces tensors in `HWC` format

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
```text

The script calls `make_calib_man()` to generate the tensor dataset:

```python
make_calib_man(
    pre_ftn=pre_ftn,
    data_dir=args.source_path,
    save_dir=os.path.dirname(args.npy_path),
    save_name=os.path.basename(args.npy_path),
    remove_npy=True,  # Clean the destination before writing new .npy files.
)
```text

Run the conversion script:

```bash
python convert_img_to_tensor.py
```text

By default, it reads images from `./coco-selected` and writes tensor files under `./calib_data_tensor`.

## Step 3: Compile the Model

Before compiling, review the required preprocessing. YOLO models typically use the `LetterBox` operation, as described in the [Ultralytics repository](https://github.com/ultralytics/ultralytics).

The Mobilint compilation API performs this preprocessing internally and can fuse the operations into the MXQ model for better NPU efficiency.

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
```text

As part of normalization, the `letterbox` step includes `1/255` scaling. You can fuse this preprocessing into the MXQ model with `Uint8InputConfig`.

When preprocessing fusion is enabled, set the MXQ input type to `uint8`:

```python
# ONNX -> MXQ: quantized package that runs on the NPU
mxq_compile(
    # ... model, calibration data, backend, and target device settings
    preprocessing_config=preprocessing_config,
    uint8_input_config=Uint8InputConfig(apply=True, inputs=[]),
    calibration_config=calibration_config,
)
```text

If you want to keep the original input format, disable both preprocessing fusion and `Uint8InputConfig`.

The example also uses the following quantization configuration:

```python
calibration_config = CalibrationConfig(
    method=1,  # 0 for per tensor, 1 for per channel
    output=1,  # 0 for layer, 1 for channel
    mode=1,  # maxpercentile
    max_percentile={
        "percentile": 0.9999,  # quantization percentile
        "topk_ratio": 0.01,  # quantization topk
    },
)
```text

After configuring the options, run `model_compile.py` with `--target-device` set for your hardware. The script generates both outputs in one run: the quantized MXQ file (`--save-path`) and the intermediate MBLT graph (`--mblt-path`).

## Step 3-1 (Optional): Compile with Prepared Tensor Files

If you already generated `.npy` tensor files, you can use that directory as `calib_data_path` instead of providing raw image files and a preprocessing pipeline.

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

- `--onnx-path`: path to the ONNX model
- `--calib-data-path`: path to the calibration data
- `--save-path`: path to save the MXQ model (`onnx -> mxq`)
- `--mblt-path`: path to save the MBLT intermediate graph (`onnx -> mblt`)
- `--target-device` (required): target NPU. See the table below. The inference scheme is chosen automatically (`ARIES = all`, `REGULUS = single`).

Outputs:

- MXQ model at `--save-path` (`onnx -> mxq`, quantized NPU package)
- MBLT intermediate graph at `--mblt-path` (`onnx -> mblt`, pre-quantization graph)

### Select the Target Device (`--target-device`)

The required model depends on the target device:

- Older REGULUS hardware (`regulus-ra`, customers before 2026-06) supports YOLOv9 and earlier, so it uses a YOLOv8 segmentation model.
- ARIES (`aries-rb`) and newer REGULUS hardware (`regulus-rb`, customers from 2026-06) use the YOLO11 segmentation model.

| User | `--target-device` | Model |
|---|---|---|
| ARIES | `aries-rb` | `yolo11m-seg` |
| REGULUS (customers before 2026-06) | `regulus-ra` | `yolov8m-seg` |
| REGULUS (customers from 2026-06) | `regulus-rb` | `yolo11m-seg` |

Export the matching model first. Step 1 shows `yolo11m-seg`; for `regulus-ra`, export `yolov8m-seg` instead:

```bash
# YOLO11 seg (for aries-rb / regulus-rb)
yolo export model=yolo11m-seg.pt format=onnx

# YOLOv8 seg (for regulus-ra)
yolo export model=yolov8m-seg.pt format=onnx
```text

```bash
# ARIES
python model_compile.py --onnx-path ./yolo11m-seg.onnx --calib-data-path ./coco-selected --save-path ./yolo11m-seg.mxq --mblt-path ./yolo11m-seg.mblt --target-device aries-rb

# REGULUS (customers before 2026-06)
python model_compile.py --onnx-path ./yolov8m-seg.onnx --calib-data-path ./coco-selected --save-path ./yolov8m-seg.mxq --mblt-path ./yolov8m-seg.mblt --target-device regulus-ra

# REGULUS (customers from 2026-06)
python model_compile.py --onnx-path ./yolo11m-seg.onnx --calib-data-path ./coco-selected --save-path ./yolo11m-seg.mxq --mblt-path ./yolo11m-seg.mblt --target-device regulus-rb
```text

After the command completes, the corresponding MXQ and MBLT files are saved in the current directory.
