# Face Detection Model Compilation

This tutorial provides detailed instructions for compiling face detection models using the Mobilint qb compiler.

In this tutorial, we will use the [YOLO11m-face](https://github.com/YapaLab/yolo-face) model, which is pretrained on the WiderFace dataset.

## Prerequisites

Before starting, ensure you have the following installed:

- qubee SDK compiler installed (version >= 0.11 required)

## Overview

The compilation process typically consists of three main steps:

1. **Model Preparation**: Download the model and export it to ONNX format
2. **Calibration Dataset Generation**: Create calibration data from a suitable dataset
3. **Model Compilation**: Convert the model to `.mxq` format using calibration data
