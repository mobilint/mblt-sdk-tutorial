# Face Detection Runtime

This tutorial explains how to run a compiled face detection MXQ model with Mobilint `qbruntime`.

Before starting, complete the compilation flow in [../../../compilation/face_detection/README.md](../../../compilation/face_detection/README.md). The runtime example in this directory expects the compiled model at `../../../compilation/face_detection/yolov12m-face.mxq`.

The overall runtime structure is similar to [../object_detection/README.md](../object_detection/README.md), but this tutorial uses a single-class face detector and a dedicated postprocess pipeline.

## Prerequisites

Make sure the following components are available:

- Mobilint `qbruntime`
- A compiled `.mxq` face detection model
- Python packages: `opencv-python`, `numpy`, `torch`

If the Python packages are not already installed in your environment, install them with:

```bash
pip install opencv-python numpy torch
```

## Overview

The runtime flow is implemented in `inference_mxq.py` and follows these steps:

1. Load the compiled MXQ model with `qbruntime`.
2. Read the input image and apply YOLO-style letterbox preprocessing.
3. Match the model input layout automatically, whether the model expects HWC or CHW input.
4. Run inference on the Mobilint NPU.
5. Decode single-class face detections, apply NMS, and render face boxes with confidence scores.

The compiled MXQ model already includes `/255` normalization, so this example keeps the runtime input in `uint8` format.

## Files in This Tutorial

- `inference_mxq.py`: Runs the full inference pipeline and saves the rendered result.
- `postprocess.py`: Rearranges face-detection outputs, decodes anchorless predictions, and applies NMS.
- `visualize.py`: Draws face bounding boxes and confidence scores on the source image.
- `face_metadata.py`: Defines the single `face` label and visualization metadata.
- `utils.py`: Contains helper functions used by postprocessing.

## How the Script Works

The script first initializes the accelerator and launches the compiled model:

```python
acc = qbruntime.Accelerator()
mc = qbruntime.ModelConfig()
mc.set_single_core_mode(None, [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)])

model = qbruntime.Model(args.model_path, mc)
model.launch(acc)
```

Next, it reads the image, converts BGR to RGB, and applies letterbox preprocessing. The code checks the model input shape and automatically prepares either HWC or CHW input as required by the compiled model.

```python
def preprocess_yolo(img: np.ndarray, input_shape) -> np.ndarray:
    if input_shape[-1] == 3:
        target_h, target_w, is_hwc = input_shape[0], input_shape[1], True
    else:
        target_h, target_w, is_hwc = input_shape[1], input_shape[2], False

    ...
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img = np.transpose(img, (2, 0, 1)) if not is_hwc else img
    # img /= 255.0  # Apply 1/255 normalization when the model expects float32 input in [0, 1].
    return np.ascontiguousarray(np.expand_dims(img, 0))
```

After inference, the script converts HWC-style NPU outputs back to BCHW when needed so that `postprocess.py` can process a single layout. `YoloPostProcessAnchorless` then groups box and classification heads, decodes distribution focal loss (DFL) predictions, and applies Non-Maximum Suppression (NMS).

Because this is a single-class model, every retained detection is labeled as `face`.

You can inspect the `.mblt` file generated during compilation in [Mobilint Netron](https://netron.mobilint.com/) if you want to confirm the output tensors and postprocessing assumptions.

## Run the Example

Run the tutorial with the default sample paths:

```bash
python inference_mxq.py
```

This command uses the following defaults:

- Model: `../../../compilation/face_detection/yolov12m-face.mxq`
- Input image: `../rc/cr7.jpg`
- Output image: `./tmp/cr_demo.jpg`

To pass the paths explicitly or adjust the thresholds, run:

```bash
python inference_mxq.py --model-path ../../../compilation/face_detection/yolov12m-face.mxq --image-path ../rc/cr7.jpg --output-path ./tmp/cr_demo.jpg --conf-thres 0.25 --iou-thres 0.45
```

## Parameters

- `--model-path`: Path to the compiled `.mxq` model.
- `--image-path`: Path to the input image.
- `--output-path`: Path to save the visualized output image.
- `--conf-thres`: Confidence threshold used to keep detections. Default: `0.25`.
- `--iou-thres`: IoU threshold used during NMS. Default: `0.45`.

## Expected Output

The script saves a rendered result image such as `tmp/cr_demo.jpg` with face bounding boxes and confidence scores overlaid on the original image.
