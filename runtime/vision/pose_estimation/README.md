# Pose Estimation Model Inference

This tutorial provides step-by-step instructions for running inference with compiled pose estimation models using the Mobilint qbruntime.

This guide is a continuation of [mblt-sdk-tutorial/compilation/vision/pose_estimation/README.md](file:///workspace/mblt-sdk-tutorial/compilation/vision/pose_estimation/README.md). It is assumed that you have successfully compiled the model and have the following file ready:

- `./yolo11m-pose.mxq` - Compiled model file

## Prerequisites

Before running inference, ensure you have the following components installed and available:

- `qbruntime` library (to access the NPU accelerator)
- Compiled `.mxq` model file
- Python packages: `opencv-python`, `numpy`, `torch`

## Overview

The inference logic is implemented in the `inference_mxq.py` script. This script demonstrates the following workflow:

1.  **Model Loading**: Load the compiled `.mxq` model via `qbruntime`.
2.  **Preprocessing**: Prepare the input image (e.g., resize with letterboxing).
3.  **Inference**: Execute the model on the NPU accelerator.
4.  **Postprocessing**: Process the model output (decode bounding box coordinates and keypoints, apply non-maximum suppression).
5.  **Visualization**: Draw bounding boxes, labels, and skeleton connections on the original image.

To better understand which operations are required for postprocessing, you can inspect the `.mblt` file (generated during compilation) using [Mobilint Netron](https://netron.mobilint.com/).

## Running Inference

The `inference_mxq.py` script performs inference in several detailed steps.

First, initialize the NPU accelerator and the model configuration.

```python
acc = qbruntime.Accelerator(0)
mc = qbruntime.ModelConfig()
mc.set_single_core_mode(1)
mxq_model = qbruntime.Model(args.mxq_path, mc)
mxq_model.launch(acc)
```

Next, load and preprocess the input image. Since the normalization operation is fused into the MXQ model during compilation, the input image should remain in `UInt8` format.

```python
def preprocess_yolo(img_path: str, img_size=(640, 640)):
    # Reference: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py#L1535
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]  # Original height and width
    r = min(img_size[0] / h0, img_size[1] / w0)  # Scale ratio
    new_unpad = int(round(w0 * r)), int(round(h0 * r))

    if (w0, h0) != new_unpad:  # Resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    dh, dw = img_size[0] - new_unpad[1], img_size[1] - new_unpad[0]  # Width and height padding
    dw /= 2  # Divide padding for both sides
    dh /= 2  # To center the image
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # Add border padding

    return img
```

Finally, execute the model with the preprocessed input and apply postprocessing to interpret the results.

To run the example inference script, use the following command:

```bash
python inference_mxq.py --model-path ../../../compilation/vision/pose_estimation/yolo11m-pose.mxq --image-path ../rc/cr7.jpg --output-path tmp/cr7.jpg --conf-thres 0.25 --iou_thres 0.45
```

### Script Breakdown

- **Model Execution**: Loads the `.mxq` file onto the NPU.
- **Preprocessing**: Resizes the image to 640x640 while collecting aspect ratio (letterboxing), pads with gray borders, and keeps data in the appropriate format.
- **Inference**: Runs the forward pass on the NPU.
- **Postprocessing**: decodes the raw output into bounding boxes and keypoints, filters by confidence score, and applies Non-Maximum Suppression (NMS).
- **Visualization**: Overlays the detected bounding boxes, class labels, keypoints, and skeleton connections onto the output image.

### Parameters

- `--model-path`: Path to the compiled `.mxq` model file.
- `--image-path`: Path to the input image file.
- `--output-path`: (Optional) Path where the output image will be saved. Defaults to `output.jpg` in the current directory if not specified.
- `--conf-thres`: Confidence threshold for filtering detections (default: `0.25`).
- `--iou-thres`: IoU (Intersection over Union) threshold for NMS (default: `0.45`).

### Expected Output

The script prints detection results (labels and confidence scores) to the console and saves an image with drawn bounding boxes and keypoints to `tmp/cr7.jpg`.
