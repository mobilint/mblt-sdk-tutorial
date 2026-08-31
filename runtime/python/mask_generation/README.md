# Mask Generation Runtime

This tutorial explains how to run the compiled `SAM2 Hiera large` MXQ models with Mobilint `qbruntime`.

Before starting, complete the compilation flow in [../../../compilation/mask_generation/README.md](../../../compilation/mask_generation/README.md). The runtime example in this directory expects the compiled models at `../../../compilation/mask_generation/sam2_hiera_large_encoder.mxq` and `../../../compilation/mask_generation/sam2_hiera_large_decoder.mxq`.

## Prerequisites

Make sure the following components are available:

- Mobilint `qbruntime`
- Both compiled `.mxq` model files
- A local checkout of [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- Python packages: `torch`, `numpy`, `pillow`

If the Python packages are not already installed in your environment, install them with:

```bash
pip install -r requirements.txt
```

SAM2 itself is not on PyPI, so install it from the official repository:

```bash
git clone https://github.com/facebookresearch/sam2.git /workspace/sam2
pip install -e /workspace/sam2
```

If you prefer not to install it, clone it anywhere and pass the path with `--sam2-root` instead.

The SAM2 checkpoint is downloaded from Hugging Face on first use, so the runtime host needs network access or a warm Hugging Face cache.

## Overview

SAM2 is promptable: the image encoder runs once per image, and the mask decoder runs once per prompt. Only those two stages are compiled. The rest of the model stays on the host and runs with official SAM2 code.

The runtime flow is implemented in `inference_mxq.py` and follows these steps:

1. Load both compiled MXQ models with `qbruntime` and pin each to its own core.
2. Apply the official SAM2 image transform to produce a `[1024, 1024, 3]` float32 input.
3. Run the encoder MXQ on the Mobilint NPU to obtain three FPN feature levels.
4. Install those features into the host predictor and run the prompt encoder.
5. Feed the six assembled decoder inputs to the decoder MXQ.
6. Upscale the mask logits to the original image size and render overlays.

```text
image
  -> SAM2 image transform                     host
  -> image encoder                            encoder MXQ
  -> prompt encoder and token assembly        host
  -> mask decoder body                        decoder MXQ
  -> mask upscaling                           host
```

## Files in This Tutorial

- `inference_mxq.py`: Runs the full two-model pipeline and saves overlays and raw outputs.
- `sam2_host.py`: Host-side SAM2 helpers for preprocessing, feature installation, prompt encoding, and mask upscaling.
- `contracts.py`: Decoder input ordering, shape validation, and decoder output identification.
- `visualize.py`: Renders each mask candidate with the prompt points drawn on top.
- `requirements.txt`: Python dependencies for the host SAM2 code.

## How the Script Works

### Launching Two Models

Both models are resident at the same time, so each is pinned to its own core instead of letting both claim the same one:

```python
def launch_model(path: str, accelerator: qbruntime.Accelerator, core: qbruntime.Core) -> qbruntime.Model:
    model_config = qbruntime.ModelConfig()
    model_config.set_single_core_mode(None, [qbruntime.CoreId(qbruntime.Cluster.Cluster0, core)])
    model = qbruntime.Model(path, model_config)
    model.launch(accelerator)
    return model


accelerator = qbruntime.Accelerator()
encoder = launch_model(args.encoder_mxq, accelerator, qbruntime.Core.Core0)
decoder = launch_model(args.decoder_mxq, accelerator, qbruntime.Core.Core1)
```

A single `Accelerator` is created once and kept in scope until both models are disposed, rather than being passed as a temporary to each `launch` call.

### Batch Dimension

The SAM2 host code produces tensors with an outer batch axis, but `qbruntime` omits that axis from its buffer shapes. Every feed is therefore stripped before inference:

```python
def strip_runtime_batch(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim >= 4 and value.shape[0] == 1:
        value = value[0]
    return np.ascontiguousarray(value, dtype=np.float32)
```

This is the opposite of the single-input vision tutorials in this repository, which add a batch dimension with `np.expand_dims` before calling `infer`. Do not copy that pattern here.

### Encoder Outputs

The encoder returns three FPN levels. The script identifies them by channel count rather than by position, so it works whether the runtime reports NHWC or NCHW:

```python
if array.shape[-1] in (32, 64, 256):
    channel = int(array.shape[-1])
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)[None]
elif array.shape[0] in (32, 64, 256):
    channel = int(array.shape[0])
    tensor = torch.from_numpy(np.ascontiguousarray(array))[None]
```

The three levels are then installed into the host predictor, which skips its own encoder and uses the NPU features for prompt encoding and mask upscaling.

### Decoder Input Ordering

This is the easiest part of the pipeline to get wrong. The decoder has six inputs and three of them share the shape `(1, 256, 64, 64)`, so position alone cannot tell them apart.

Runtime positional order, used here:

```text
image_embeddings, dense_prompt_embeddings, image_pe, sparse_prompt_embeddings, hrf0_nhwc, hrf1_nhwc
```

This matches the MBLT input-name order used during calibration and compilation, so there is only one order to keep straight. Confirm it against your own artifact without an NPU:

```bash
python -c "import qbruntime; print(qbruntime.get_model_summary('../../../compilation/mask_generation/sam2_hiera_large_decoder.mxq'))"
```

which reports:

```text
Input - Shapes: [(256, 64, 64), (256, 64, 64), (256, 64, 64), (1, -1, 256),
                 (256, 256, 32), (128, 128, 64)]
```

The `-1` is the prompt axis, so one compiled decoder serves any point count. Feeding the tensors in the wrong order produces plausible but wrong masks rather than an error, so `contracts.py` builds the feed by semantic role and then shape-checks it against the runtime:

```python
decoder_feed = build_decoder_runtime_feed(decoder_tensors, args.decoder_runtime_order)
validate_runtime_shapes(decoder_feed, decoder.get_model_input_shape(), "decoder")
```

If you recompile with a different decoder MBLT, confirm the runtime order with `get_model_summary` and pass it through `--decoder-runtime-order`.

### Decoder Outputs

The decoder is parsed with `output_meta=lambda x: x[0][:2]`, so it exposes two outputs: masks and IoU. Older wrapper-traced decoders also emitted SAM tokens and an object score, and those are still accepted when present.

qbruntime does not guarantee that the runtime output order matches the compiled graph's declared order, so outputs are identified by element count instead of position: the mask output is a multiple of `256 x 256`, the IoU scores match the mask count, the SAM tokens are `num_masks x 256`, and the object score is a single value. Every output is also checked for NaN and infinity, because a non-finite value would otherwise pass silently through the IoU `argmax` and the `> 0` mask threshold and corrupt the prediction rather than report a failure.

Only masks and IoU affect the segmentation result. The script reports all three mask candidates and marks the one with the highest predicted IoU as `selected`.

## Run the Example

The prompt is required, because SAM2 segments whatever the prompt points at. Points use original image coordinates in `X,Y,LABEL` form, where `1` is a positive point and `0` is a negative point:

```bash
python inference_mxq.py --point 500,580,1
```

This command uses the following defaults:

- Encoder model: `../../../compilation/mask_generation/sam2_hiera_large_encoder.mxq`
- Decoder model: `../../../compilation/mask_generation/sam2_hiera_large_decoder.mxq`
- Input image: `../rc/bus.jpg`
- Output directory: `./tmp/demo`

To pass the paths explicitly, or to combine positive and negative points, run:

```bash
python inference_mxq.py --encoder-mxq ../../../compilation/mask_generation/sam2_hiera_large_encoder.mxq --decoder-mxq ../../../compilation/mask_generation/sam2_hiera_large_decoder.mxq --image-path ../rc/bus.jpg --output-dir ./tmp/custom --point 500,580,1 --point 400,120,0
```

The decoder accepts one to three points. The compiled model supports this range because calibration used a mixed point count, which marks the token axis as dynamic.

## Parameters

- `--encoder-mxq`: Path to the compiled encoder MXQ model.
- `--decoder-mxq`: Path to the compiled decoder MXQ model.
- `--image-path`: Path to the input image. Default: `../rc/bus.jpg`.
- `--point`: Prompt point as `X,Y,LABEL` in original image coordinates. Repeat for up to three points. Required.
- `--output-dir`: Directory for overlays and raw outputs. Default: `./tmp/demo`.
- `--sam2-root`: Local `facebookresearch/sam2` checkout.
- `--model-id`: SAM2 model id. Default: `facebook/sam2-hiera-large`.
- `--torch-device`: Torch device for the host SAM2 code. Defaults to `cuda` when available, otherwise `cpu`.
- `--decoder-runtime-order`: Comma-separated semantic input order reported by `mxq_show`.

## Expected Output

The output directory contains:

- `mask_0.png`, `mask_1.png`, `mask_2.png`: the three mask candidates overlaid on the original image, with prompt points drawn
- `outputs.npz`: binary masks, full-resolution logits, low-resolution logits, predicted IoU, and the selected index, plus SAM tokens and object score on a four-output decoder
- `summary.json`: the prompt, the predicted IoU per candidate, and the selected candidate index

## Notes

- This tutorial targets the SAM2 Hiera decoder contract with `use_multimask_token_for_obj_ptr=True`. Other SAM2 decoder variants are not supported.
- Box prompts and mask prompts are not supported; only point prompts are.
- The host SAM2 model is still loaded in full, because the prompt encoder and the mask upscaling run on the host. The script selects CUDA when it is available and otherwise falls back to CPU, which is slower but works on a CPU-only runtime host.
- Core pinning replaces the deprecated `set_single_core_mode(num_cores=...)` form. The accuracy figures quoted in the compilation guide were measured with the earlier launch configuration.
- You can inspect the `.mblt` files used during compilation in [Mobilint Netron](https://netron.mobilint.com/) if you want to confirm the input and output tensors.
- Full execution requires a working Mobilint runtime environment and compatible hardware.
