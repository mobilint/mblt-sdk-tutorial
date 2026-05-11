# Face Detection Model Inference

This tutorial explains how to run inference with a compiled face detection model using Mobilint `qbruntime`.

The structure follows [../object_detection/README.md](../object_detection/README.md), but the postprocessing and labels are adapted for a single-class face detector.

This guide continues from [../../compilation/face_detection/README.md](../../compilation/face_detection/README.md). It assumes you already have a compiled model such as:

- `../../compilation/face_detection/yolov12m-face.mxq`

## Prerequisites

Before running inference, make sure the following are available:

- `qbruntime`
- A compiled `.mxq` face detection model
- Python packages: `opencv-python`, `numpy`, `torch`

Install the Python packages with:

```bash
pip install opencv-python numpy torch
```

## Overview

The inference pipeline is implemented in `inference_mxq.py` and follows these stages:

1. **Model Loading**: Load the compiled `.mxq` model with `qbruntime`.
2. **Preprocessing**: Read the input image and apply the same `640x640` letterbox preprocessing used during compilation.
3. **Inference**: Run the model on the Mobilint NPU.
4. **Postprocessing**: Rearrange YOLO heads, decode anchorless predictions, and apply NMS.
5. **Visualization**: Draw detected faces and confidence scores on the original image.

To inspect the compiled graph and confirm the model outputs, you can open the `.mblt` file in [Mobilint Netron](https://netron.mobilint.com/).

## Running Inference

The script first initializes the accelerator and model configuration:

```python
acc = qbruntime.Accelerator()
mc = qbruntime.ModelConfig()
mc.set_single_core_mode(None, [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)])

model = qbruntime.Model(args.model_path, mc)
model.launch(acc)
```

Next, the input image is loaded and letterboxed. Because normalization is fused into the compiled model, the runtime input remains in `UInt8` format.

```python
def preprocess_yolo(img_path: str, img_size: tuple[int, int] = (640, 640)) -> np.ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ...
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(img)
```

The raw outputs are then passed to `YoloPostProcessAnchorless`, which:

- Splits NPU outputs into box heads and classification heads
- Decodes distribution focal loss (DFL) box predictions
- Filters detections by confidence threshold
- Applies Non-Maximum Suppression (NMS)

Finally, `YoloVisualizer` rescales the boxes back to the original image size and writes the visualized result.

Run the example with:

```bash
python inference_mxq.py --model-path ../../compilation/face_detection/yolov12m-face.mxq --image-path ../rc/cr7.jpg --output-path ./tmp/cr_demo.jpg --conf-thres 0.25 --iou-thres 0.45
```

This example passes the model path explicitly so it matches the compiled artifact from the compilation tutorial.

## Parameters

- `--model-path`: Path to the compiled `.mxq` model file
- `--image-path`: Path to the input image
- `--output-path`: Path where the visualized result image will be saved
- `--conf-thres`: Confidence threshold for filtering detections
- `--iou-thres`: IoU threshold used during NMS

## Expected Output

The script saves a result image such as `tmp/cr_demo.jpg` with face bounding boxes drawn on the original image.

Because this is a single-class detector, every kept detection is labeled as `face`.
