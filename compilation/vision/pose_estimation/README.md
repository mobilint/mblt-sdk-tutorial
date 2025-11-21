# Pose Estimation

This tutorial provides detailed instructions for compiling pose estimation models using the Mobilint qb compiler.

In this tutorial, we will use the [YOLO11m-pose](https://docs.ultralytics.com/models/yolo11/) model, which is pretrained on the COCO dataset developed by Ultralytics. This model is a pose estimation model that can be used to estimate the pose of objects in images.

## Prerequisites

Before starting, ensure you have the following installed:

- qubee SDK compiler installed (version >= 0.11 required)

Also, you need to install the following packages:

```bash
pip install ultralytics
```

## Overview

The compilation process consists of three main steps:

1. **Model Preparation**: Download the model and export it to ONNX format
2. **Calibration Dataset Generation**: Create calibration data from COCO dataset
3. **Model Compilation**: Convert the model to `.mxq` format using calibration data

## Step 1: Model Preparation

First, we need to prepare the model. We will use the `ultralytics` library to download the pretrained model and export it to ONNX format.

```bash
yolo export model=yolo11m-pose.pt format=onnx # Export the model to ONNX format
```

After execution, the exported ONNX model is saved as `yolo11m-pose.onnx` in the current directory.

## Step 2: Calibration Dataset Preparation

## Step 3: Model Compilation
