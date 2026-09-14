# Vision-Language Model Runtime

This tutorial runs the prepared `Qwen3-VL-2B-Instruct` model with `mblt-model-zoo`.

## Prerequisites

```bash
pip install -r requirements.txt
```

First complete the [VLM compilation tutorial](../../../compilation/vlm/README.md), including `prepare_model.py`.

## Run Inference

The default command uses the ARIES model prepared at `compilation/vlm/prepared/aries-rb/Qwen3-VL-2B-Instruct`.

```bash
python inference_mblt_model_zoo.py
```

You can also provide a local image or URL and a prompt.

```bash
python inference_mblt_model_zoo.py \
  --image /path/to/image.jpg \
  --prompt "What objects are visible in this image?"
```
