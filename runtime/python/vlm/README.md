# Vision-Language Model Runtime

This tutorial runs the prepared `Qwen3-VL-2B-Instruct` model with `mblt-model-zoo`.

## Prerequisites

```bash
pip install -r requirements.txt
```

First complete the [VLM compilation tutorial](../../../compilation/vlm/README.md), including `prepare_model.py`.

## Run Inference

The default command uses the dynamic ARIES model prepared at `compilation/vlm/prepared/aries-rb/Qwen3-VL-2B-Instruct`.

```bash
python inference_mblt_model_zoo.py
```

To run a static build prepared with `prepare_model.py --static`, pass its folder:

```bash
python inference_mblt_model_zoo.py \
  --model-folder ../../../compilation/vlm/prepared/aries-rb/Qwen3-VL-2B-Instruct-static
```

The static build needs a `transformers-mblt` release that contains the fix for the folded static encoder (PR link: **TBD**).
<!-- TODO: add the transformers-mblt PR link for the folded static-encoder fix. -->
The dynamic build runs with the `mblt-model-zoo` version in `requirements.txt`.

You can also provide a local image or URL and a prompt.

```bash
python inference_mblt_model_zoo.py \
  --image /path/to/image.jpg \
  --prompt "What objects are visible in this image?"
```
