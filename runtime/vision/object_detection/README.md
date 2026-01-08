# Object Detection Model Inference

This tutorial provides detailed instructions for running inference with compiled object detection models using the Mobilint qb runtime.

This guide continues from `mblt-sdk-tutorial/compilation/vision/object_detection/README.md`. We assume you have successfully compiled the model and have the following files ready:

- `./yolo11m.mxq` - Compiled model file

## Prerequisites

Before running inference, ensure you have:

- maccel runtime library (provides NPU accelerator access)
- Compiled `.mxq` model file
- Python packages: `opencv-python`, `numpy`, `torch`

## Overview

The inference process is implemented in the `inference_mxq.py` script. This script demonstrates how to:

- Load the compiled `.mxq` model using maccel runtime
- Preprocess the input image (resize, pad, normalize)
- Run inference on the NPU accelerator
- Postprocess the output (convert to bounding box coordinates, apply non-max suppression)
- Visualize the results

To analyze which operations should be included in the postprocessing, you may upload the `.mblt` file, which is generated during the compilation process, to [Mobilint Netron](https://netron.mobilint.com/).

## Running Inference

To run the example inference script:

```bash
python inference_mxq.py --model_path ../../../compilation/vision/object_detection/yolo11m.mxq --image_path ../rc/cr7.jpg --output_path tmp/cr7.jpg --conf_thres 0.25 --iou_thres 0.45
```

**What this does:**

- Loads the compiled `.mxq` model onto the NPU accelerator
- Loads and preprocesses the input image (YOLO preprocessing: resize to 640x640 with aspect ratio preservation, pad with gray borders, normalize to [0, 1])
- Runs inference on the NPU accelerator
- Postprocesses the output (converts raw predictions to bounding box coordinates, filters detections by confidence threshold, applies non-maximum suppression)
- Visualizes the results by drawing bounding boxes and labels on the image

**Parameters:**

- `--model_path`: Path to the compiled `.mxq` model file
- `--image_path`: Path to the input image
- `--output_path`: Path to save the output image (optional, defaults to `output.jpg` in the same directory as input)
- `--conf_thres`: Confidence threshold for filtering detections (default: 0.25)
- `--iou_thres`: IoU threshold for non-maximum suppression (default: 0.45)

**Expected output:**

The script will display the detection results with their confidence scores and class labels, and save the output image with bounding boxes drawn in the `tmp/` directory with the name `cr7.jpg`.
