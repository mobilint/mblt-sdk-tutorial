# Pose Estimation

This tutorial provides detailed instructions for compiling object detection models using the Mobilint qb compiler.

In this tutorial, we will use the [YOLO11m-pose](https://docs.ultralytics.com/models/yolo11/) model, a pose estimation model developed by Ultralytics.

## Model Preparation

First, we need to prepare the model. We will use the `ultralytics` library to download the pretrained model and export it to ONNX format.

```bash
pip install ultralytics # Install the ultralytics library if not installed
yolo export model=yolo11m-pose.pt format=onnx # Export the model to ONNX format
```

After execution, the exported ONNX model is saved as `yolo11m-pose.onnx` in the current directory.

## Calibration Dataset Preparation



## Model Compilation