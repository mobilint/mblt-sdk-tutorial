# Object Detection Model Compilation

This tutorial shows how to compile an object detection model with Mobilint `qbcompiler`.

The example uses [YOLO11m](https://docs.ultralytics.com/models/yolo11/), pretrained on the COCO dataset by Ultralytics. The model detects and localizes multiple objects in an image.

## Prerequisites

Before you begin, make sure you have:

- `qbcompiler`
- A Hugging Face account with access to the COCO dataset

Install the required Python packages:

```bash
pip install ultralytics aiohttp aiofiles
```

## Overview

The workflow has three main steps:

1. **Prepare the model**: Download the model and export it to ONNX.
2. **Prepare the calibration dataset**: Build a representative calibration dataset from COCO.
3. **Compile the model**: Convert the model to `.mxq` using the calibration data.

## Step 1: Prepare the Model

Use `ultralytics` to download the pretrained model and export it to ONNX:

```bash
yolo export model=yolo11m.pt format=onnx
```

After the command finishes, the exported model is saved as `yolo11m.onnx` in the current directory.

## Step 2: Prepare the Calibration Dataset

The calibration dataset should represent the model's expected input distribution. Because YOLO11m is trained on the [COCO dataset](https://cocodataset.org/#download), this tutorial uses COCO samples for calibration.

Before downloading the dataset, sign in to Hugging Face with your token:

```bash
hf auth login --token <your_huggingface_token>
```

If you need to create or locate your token, see [Hugging Face account settings](https://huggingface.co/settings/tokens).

Run `prepare_coco.py` to download a randomly selected set of COCO images into `coco-selected`:

```bash
python prepare_coco.py
```

This script:

- Downloads COCO image URLs from Hugging Face
- Randomly selects images for calibration
- Saves the downloaded images to `coco-selected`

Output:

- `coco-selected`: calibration dataset directory

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

Before compiling, confirm the required preprocessing steps. YOLO models typically use the `LetterBox` operation, as described in the [Ultralytics repository](https://github.com/ultralytics/ultralytics).

The Mobilint compilation API applies the preprocessing pipeline during calibration. The normalization step (`/255` scaling) is fused into the MXQ model through `Uint8InputConfig`, so the runtime model can accept `uint8` input directly. Spatial transforms such as letterboxing are not fused and must still be applied at runtime.

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
```

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

The recommended model depends on the target device:

- `regulus-ra` supports YOLOv9 and earlier, so use `yolov9m`
- `aries-rb` and `regulus-rb` use `yolo11m`

| User | `--target-device` | Model |
| --- | --- | --- |
| ARIES | `aries-rb` | `yolo11m` |
| REGULUS (customers before 2026-06) | `regulus-ra` | `yolov9m` |
| REGULUS (customers from 2026-06) | `regulus-rb` | `yolo11m` |

Export the matching model first. Step 1 shows `yolo11m`; for `regulus-ra`, export `yolov9m` instead:

```bash
# YOLO11 (for aries-rb / regulus-rb)
yolo export model=yolo11m.pt format=onnx

# YOLOv9 (for regulus-ra)
yolo export model=yolov9m.pt format=onnx
```

Then run the compiler:

```bash
# ARIES
python model_compile.py --onnx-path ./yolo11m.onnx --calib-data-path ./coco-selected --save-path ./yolo11m.mxq --mblt-path ./yolo11m.mblt --target-device aries-rb

# REGULUS (customers before 2026-06)
python model_compile.py --onnx-path ./yolov9m.onnx --calib-data-path ./coco-selected --save-path ./yolov9m.mxq --mblt-path ./yolov9m.mblt --target-device regulus-ra

# REGULUS (customers from 2026-06)
python model_compile.py --onnx-path ./yolo11m.onnx --calib-data-path ./coco-selected --save-path ./yolo11m.mxq --mblt-path ./yolo11m.mblt --target-device regulus-rb
```

After the command completes, the corresponding MXQ and MBLT files are saved in the current directory.
