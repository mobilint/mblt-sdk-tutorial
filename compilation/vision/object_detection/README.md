# Object Detection

This tutorial provides detailed instructions for compiling object detection models using the Mobilint qb compiler.

In this tutorial, we will use the [YOLO11m](https://docs.ultralytics.com/models/yolo11/) model, an object detection model developed by Ultralytics.

## Model Preparation

First, we need to prepare the model. We will use the `ultralytics` library to download the pretrained model and export it to ONNX format.

```bash
pip install ultralytics # Install the ultralytics library if not installed
yolo export model=yolo11m.pt format=onnx # Export the model to ONNX format
```

After execution, the exported ONNX model is saved as `yolo11m.onnx` in the current directory.

## Calibration Dataset Preparation

The YOLO11m model is trained on the COCO dataset, so we need to prepare the calibration dataset.

```bash
wget http://images.cocodataset.org/zips/val2017.zip # Download the validation dataset
unzip val2017.zip # Unzip the dataset
```

> Note: According to the [COCO dataset](https://cocodataset.org/#download) page, downloading the dataset through Google Cloud Platform is recommended, but currently it is not available.

The calibration dataset should be pre-processed to be compatible with the quantized model. Therfore, we should first investigate the pre-processing operation used in the original model. The pre-processing operation is defined in [Ultralytics' GitHub](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py). We wrote the simplified but equivalent operation as follows:

```python
import numpy as np
import cv2

img_size = [640, 640] 
def preprocess_yolo(img_path: str):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h0, w0 = img.shape[:2]  # original hw
    r = min(img_size[0] / h0, img_size[1] / w0)  # ratio
    new_unpad = int(round(w0 * r)), int(round(h0 * r))
    dh, dw = (
        img_size[0] - new_unpad[1],
        img_size[1] - new_unpad[0],
    )  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if (img.shape[1], img.shape[0]) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # add border
    img = (img / 255).astype(np.float32)

    return img

```

One of the qb compiler's utility functions is `make_calib_man`, which can be used to create a calibration dataset with custom pre-processing functions. The script `prepare_calib.py` uses this function to create a calibration dataset with the pre-processing operation defined above.

```bash
python3 prepare_calib.py --data_dir {path_to_calibration_dataset} --img_size {image_size} --save_dir {path_to_save_calibration_dataset} --save_name {name_of_calibration_dataset} --max_size {maximum_number_of_calibration_data}
```

The example command is as follows:

```bash
python3 prepare_calib.py --data_dir ./val2017 --img_size 640 --save_dir ./ --save_name yolo11m_cali --max_size 100
```

## Model Compilation

After the calibration dataset and the model are prepared, we can compile the model.

```bash
python3 model_compile.py --onnx_path {path_to_onnx_model} --calib_data_path {path_to_calibration_dataset} --save_path {path_to_save_model} --quant_percentile {quantization_percentile} --topk_ratio {topk_ratio} --inference_scheme {inference_scheme}
```

The quantization percentile and top-k ratio are parameters that required for running quantization algorithm.

The inference scheme is a parameter that specifies the core allocation strategy for the model. Currently, the following inference schemes are supported:

- single: Single core inference
- multi: Multi-core inference
- global: Global inference (Deprecated and replaced by global8)
- global4: Global inference with 4 cores
- global8: Global inference with 8 cores

Further details about the inference scheme can be found in the [Multi-Core Modes](https://docs.mobilint.com/v0.29/en/multicore.html) documentation.

The example command is as follows:

```bash
python3 model_compile.py --onnx_path ./yolo11m.onnx --calib_data_path ./yolo11m_cali --save_path ./yolo11m.mxq --quant_percentile 0.999 --topk_ratio 0.001 --inference_scheme single
```
