# Oriented Bounding Boxes Model Compilation

This tutorial explains how to compile an oriented bounding box (OBB) detection model with Mobilint `qbcompiler`.

The example uses [YOLO11m-obb](https://docs.ultralytics.com/tasks/obb/) from Ultralytics. The model is trained for rotated object detection on DOTA-style aerial imagery, so this tutorial uses the DOTA dataset for calibration and applies `1024x1024` letterbox preprocessing.

## Prerequisites

Before you begin, make sure you have:

- `qbcompiler`
- Python 3.10 or later

Install the required Python packages:

```bash
pip install ultralytics opencv-python
```text

## Overview

The workflow has three main steps:

1. **Prepare the model**: Download the model and export it to ONNX.
2. **Prepare the calibration dataset**: Build a representative calibration dataset from DOTA.
3. **Compile the model**: Convert the model to `.mxq` using the calibration data.

## Step 1: Prepare the Model

Use `ultralytics` to download the pretrained OBB model and export it to ONNX:

```bash
yolo export model=yolo11m-obb.pt format=onnx
```text

After the export finishes, the ONNX model is saved as `yolo11m-obb.onnx` in the current directory.

## Step 2: Prepare the Calibration Dataset

The calibration dataset should represent the model's expected input distribution. Because `yolo11m-obb.pt` is trained on the [DOTA dataset](https://captain-whu.github.io/DOTA/index.html), this tutorial uses DOTAv1 samples for calibration.

You can prepare the dataset directly from the Ultralytics archive:

```bash
python prepare_dota.py
```text

This script:

- Downloads `DOTAv1.zip` if it is not already present.
- Extracts the archive into `./DOTAv1`.
- Randomly selects 100 images with a fixed seed.
- Copies those images into `./dota-selected`.

Output:

- `dota-selected`: calibration dataset directory

### Optional Dataset Arguments

If you already downloaded the dataset manually, you can reuse it:

```bash
python prepare_dota.py --skip-download --zip-path ./DOTAv1.zip
```text

If you extracted the dataset yourself and only want to create the calibration subset:

```bash
python prepare_dota.py --skip-download --extract-dir ./DOTAv1 --output-dir ./dota-selected --num-images 100
```text

When `--skip-download` is set, an existing `--extract-dir` is reused even if it was extracted manually and does not
contain the script's `.extracted` marker file.

## Step 2-1 (Optional): Convert Images to Preprocessed Tensors

You can also prepare the calibration dataset as preprocessed `.npy` tensors. This is useful when your model needs custom preprocessing and you want to generate the tensor inputs yourself.

Since `qbcompiler` v1.0.0, the standard calibration dataset flow usually makes this step unnecessary. Use it only when you need explicit control over preprocessing.

The conversion script assumes a preprocessing function that:

- Takes an image path as input
- Returns a NumPy tensor
- Produces tensors in `HWC` format for calibration

Example preprocessing function:

```python
def pre_ftn(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h0, w0 = img.shape[:2]
    r = min(1024 / h0, 1024 / w0)
    new_unpad = int(round(w0 * r)), int(round(h0 * r))
    dh, dw = 1024 - new_unpad[1], 1024 - new_unpad[0]

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
```text

The OBB tutorial uses `1024x1024` letterboxing because that matches the exported `yolo11m-obb` model input.

The script uses `make_calib_man()` to generate the tensor dataset:

```python
make_calib_man(
    pre_ftn=pre_ftn,
    data_dir=args.source_path,
    save_dir=os.path.dirname(args.npy_path),
    save_name=os.path.basename(args.npy_path),
    remove_npy=True,
)
```text

Run the script:

```bash
python convert_img_to_tensor.py
```text

By default, it reads images from `./dota-selected` and writes the tensor dataset to `./calib_data_tensor`.

## Step 3: Compile the Model

Before compiling, confirm the required preprocessing steps. OBB models exported from Ultralytics use letterbox resizing, and this tutorial matches that behavior with a `1024x1024` letterbox operation during calibration.

The Mobilint compilation API applies this pipeline during calibration. The normalization step (`/255` scaling) is fused into the MXQ model through `Uint8InputConfig`, so the runtime model can accept `uint8` input directly. Spatial transforms such as letterboxing are not fused and must still be applied at runtime.

In `model_compile.py`, the preprocessing pipeline is defined as follows:

```python
preprocess_pipeline = [{"op": "letterbox", "height": 1024, "width": 1024, "padValue": 114}]
preprocessing_config = PreprocessingConfig(
    apply=True,
    auto_convert_format=True,
    pipeline=preprocess_pipeline,
    input_configs={},
)
```text

When preprocessing fusion is enabled, set the MXQ input type to `uint8`:

```python
mxq_compile(
    # ... model, calibration data, backend, and target device settings
    preprocessing_config=preprocessing_config,
    uint8_input_config=Uint8InputConfig(apply=True, inputs=[]),
    calibration_config=calibration_config,
)
```text

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
```text

After configuring the settings, run `model_compile.py` with `--target-device` for your hardware. A single run generates both outputs: the quantized MXQ file (`--save-path`) and the intermediate MBLT graph (`--mblt-path`).

## Step 3-1 (Optional): Compile with Prepared Tensor Files

If you already prepared `.npy` tensor files, you can use that directory as `calib_data_path` instead of supplying raw image files and a preprocessing pipeline.

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
```text

Parameters:

- `--onnx-path`: path to the ONNX model
- `--calib-data-path`: path to the calibration data
- `--save-path`: path to save the MXQ model (`onnx -> mxq`)
- `--mblt-path`: path to save the MBLT intermediate graph (`onnx -> mblt`)
- `--target-device`: target NPU. See the table below. The inference scheme is selected automatically (`ARIES = all`, `REGULUS = single`).

Outputs:

- MXQ model at `--save-path`
- MBLT intermediate graph at `--mblt-path`

### Select the Target Device (`--target-device`)

| User | `--target-device` | Model |
| --- | --- | --- |
| ARIES | `aries-rb` | `yolo11m-obb` |
| REGULUS (customers from 2026-06) | `regulus-rb` | `yolo11m-obb` |

> **Note**: OBB uses the YOLO11 `yolo11m-obb` model, which is **not supported on older REGULUS (`regulus-ra`, customers before 2026-06)** — that generation supports only YOLOv9 and earlier, and OBB is available only on later models. Use `aries-rb` or `regulus-rb`.

```bash
# ARIES
python model_compile.py --onnx-path ./yolo11m-obb.onnx --calib-data-path ./dota-selected --save-path ./yolo11m-obb.mxq --mblt-path ./yolo11m-obb.mblt --target-device aries-rb

# REGULUS (customers from 2026-06)
python model_compile.py --onnx-path ./yolo11m-obb.onnx --calib-data-path ./dota-selected --save-path ./yolo11m-obb.mxq --mblt-path ./yolo11m-obb.mblt --target-device regulus-rb
```text

After executing a command, the MXQ (`yolo11m-obb.mxq`) and MBLT (`yolo11m-obb.mblt`) are saved in the current directory.

## Files in This Tutorial

- `model_compile.py`: compiles the ONNX model into MXQ / MBLT for the selected `--target-device`
- `prepare_dota.py`: downloads or reuses DOTAv1 and prepares calibration images
- `convert_img_to_tensor.py`: converts DOTA images into preprocessed `.npy` tensors for calibration
- `README.md`: documents the end-to-end workflow for this example
