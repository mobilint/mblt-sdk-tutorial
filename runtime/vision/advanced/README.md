# Advanced Test with Image Classification Model

This tutorial provides detailed instructions for running inference with the compiled image classification models using the Mobilint qb runtime.

This guide continues from `mblt-sdk-tutorial/compilation/vision/advanced/README.md`. We assume you have successfully compiled the model and have the following files ready:

- `ILSVRC2012_bbox_val_v3.tgz`
- `ILSVRC2012_img_val.tar`

To get these files, you can use the following commands:

```bash
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
```

## Prerequisites

Before running inference, ensure you have:

- maccel runtime library (provides NPU accelerator access)
- Compiled `.mxq` model file
- Python packages: `PIL`, `numpy`, `torch`

Then, we will use `mblt-model-zoo` package's benchmark script to run inference.

Therefore, clone the repository and install the package:

```bash
git clone https://github.com/mobilint/mblt-model-zoo.git
cd mblt-model-zoo
git checkout jm/vision_bench # switch to the `jm/vision_bench` branch
pip install -e.
cd tests/vision/benchmark
```

## Organize Dataset

To use the benchmark, we first need to organize the dataset. The following command will create a directory structure that is expected by the benchmark script.

```bash
python organize_imagenet.py --image_dir ../../../../ILSVRC2012_img_val.tar --xml_dir ../../../../ILSVRC2012_bbox_val_v3.tgz
```

After running the above command, the dataset will be organized in the `~/.mblt_model_zoo/datasets/imagenet` directory.

## Run Benchmark

To run the benchmark, use the following command:

```bash
python benchmark_imagenet.py --local_path {path_to_local_mxq_file} --model_type {model_configuration_type} --infer_mode {core_allocation_mode} --batch_size {batch_size} 
```

For example, the command is as follows:

```bash
python benchmark_imagenet.py --local_path /workspace/mblt-sdk-tutorial/compilation/vision/advanced/resnet50_5cls_100_9999_01.mxq --model_type IMAGENET1K_V1 --infer_mode single --batch_size 1 
```