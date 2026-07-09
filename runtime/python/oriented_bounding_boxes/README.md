# Oriented Bounding Boxes Runtime

This tutorial explains how to run the compiled `YOLO11m-obb` MXQ model with Mobilint `qbruntime`.

Before starting, complete the compilation flow in [../../../compilation/oriented_bounding_boxes/README.md](../../../compilation/oriented_bounding_boxes/README.md). The runtime example in this directory expects the compiled model at `../../../compilation/oriented_bounding_boxes/yolo11m-obb.mxq`.

## Prerequisites

Make sure the following components are available:

- Mobilint `qbruntime`
- A compiled `.mxq` model file
- Python packages: `opencv-python`, `numpy`, `torch`

If the Python packages are not already installed in your environment, install them with:

```bash
pip install opencv-python numpy torch
```

## Overview

The runtime flow is implemented in `inference_mxq.py` and follows these steps:

1. Load the compiled MXQ model with `qbruntime`.
2. Read the input image and apply the same `1024x1024` letterbox preprocessing used during compilation.
3. Match the model input layout automatically, whether the model expects HWC or CHW input.
4. Run inference on the Mobilint NPU.
5. Decode rotated boxes, apply rotated NMS, and render polygons with DOTA class labels.

The compiled MXQ model already includes `/255` normalization through the compilation-time `Uint8InputConfig`, so this example keeps the runtime input in `uint8` format.

## Files in This Tutorial

- `inference_mxq.py`: Runs the full inference pipeline and saves the rendered result.
- `postprocess.py`: Rearranges OBB outputs, decodes `cx, cy, w, h, angle`, and applies rotated NMS.
- `visualize.py`: Converts detections back to the source image coordinates and draws rotated polygons.
- `dota.py`: Provides DOTAv1 class names and color metadata.
- `utils.py`: Contains DFL, rotated-box decode, coordinate scaling, and rotated NMS helpers.

## How the Script Works

The script first initializes the accelerator and launches the compiled model:

```python
acc = qbruntime.Accelerator()
model_config = qbruntime.ModelConfig()
model_config.set_single_core_mode(
    None,
    [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)],
)

model = qbruntime.Model(args.model_path, model_config)
model.launch(acc)
```

Next, it reads the image, converts BGR to RGB, and applies `1024x1024` letterbox preprocessing. The code checks the model input shape and automatically prepares either HWC or CHW input as required by the compiled model.

```python
def preprocess_yolo_obb(img: np.ndarray, input_shape: tuple[int, ...]) -> np.ndarray:
    if input_shape[-1] == 3:
        target_h, target_w, is_hwc = input_shape[0], input_shape[1], True
    else:
        target_h, target_w, is_hwc = input_shape[1], input_shape[2], False

    ...
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    if not is_hwc:
        img = np.transpose(img, (2, 0, 1))

    return np.expand_dims(img, 0).astype(np.uint8, copy=False)
```

After inference, the script converts HWC-style NPU outputs back to BCHW when needed so that `postprocess.py` can process a single layout. `YoloObbPostProcess` then groups detection, class, and angle heads, decodes rotated boxes, and applies rotated NMS.

The final detections follow the row format `cx, cy, w, h, conf, cls, angle`, which `visualize.py` converts to rotated polygons on the original image.

You can inspect the `.mblt` file generated during compilation in [Mobilint Netron](https://netron.mobilint.com/) if you want to confirm the output tensors and postprocessing assumptions.

## Run the Example

Run the tutorial with the default sample paths:

```bash
python inference_mxq.py
```

This command uses the following defaults:

- Model: `../../../compilation/oriented_bounding_boxes/yolo11m-obb.mxq`
- Input image: `../rc/airport.jpg`
- Output image: `./tmp/airport_demo.jpg`

To pass the paths explicitly or adjust the thresholds, run:

```bash
python inference_mxq.py --model-path ../../../compilation/oriented_bounding_boxes/yolo11m-obb.mxq --image-path ../rc/airport.jpg --output-path ./tmp/airport_custom.jpg --conf-thres 0.3 --iou-thres 0.5
```

## Parameters

- `--model-path`: Path to the compiled `.mxq` model.
- `--image-path`: Path to the input image.
- `--output-path`: Path to save the visualized output image.
- `--conf-thres`: Confidence threshold used to keep detections. Default: `0.25`.
- `--iou-thres`: IoU threshold used during rotated NMS. Default: `0.45`.

## Expected Output

The script saves a rendered result image such as `tmp/airport_demo.jpg` with rotated polygons, DOTAv1 class labels, and confidence scores overlaid on the original image.

## Notes

- This tutorial targets the `YOLO11m-obb` output layout used by the local postprocess implementation.
- The postprocess stage expects three output levels for box, class, and angle heads.
- Full execution requires a working Mobilint runtime environment and compatible hardware.
