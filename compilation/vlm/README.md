# Vision-Language Model Compilation

This tutorial compiles the encoder and decoder of [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) into MXQ models and prepares one self-contained runtime model directory.

Run all commands from `compilation/vlm`.

## Prerequisites

```bash
pip install -r requirements.txt
```

## Supported Devices

| Device | Support |
| --- | --- |
| `aries-rb` | Supported |
| `regulus-rb` | Supported |
| `regulus-ra` | Not supported |

## Vision Encoder

The encoder is compiled through qbcompiler's `vision` model part with dynamic side inputs.
The host computes the vision position embeddings and RoPE for each processed image size and supplies them to the encoder MXQ, so image sizes may vary after processor resizing and the number of image tokens follows the image size.
The decoder is compiled with a runtime RoPE input (`dynamic_rope`) so it accepts that variable number of image tokens.

> **qbcompiler 1.4 or later is required.**
> The static 224x224 encoder of earlier versions, built with `qbcompiler.model_dict_legacy` and `repreprocess_pixel_values`, has been removed together with the legacy parser.
> The encoder MXQ now takes three inputs: folded pixel values `[1, N, 1536]`, position embeddings `[1, N, 1024]`, and packed RoPE `[1, N, 128]` for the 2B model.
> `mblt-model-zoo` 2.11 or later recognizes this encoder from its input shapes.

## 1. Download Calibration Images

```bash
python download_images.py
```

The script downloads 300 COCO validation images from a fixed dataset revision, converts them to RGB, and saves them under `./images` at their original resolution.
Keeping the original sizes makes the calibration set span a range of vision patch counts N.

## 2. Generate Calibration Data

```bash
python generate_calibration_data.py --batch-size 4
```

The script creates vision encoder samples and decoder prefill/decode samples under `./calibration_data`.
The default batch size is 4 on `cuda:0`.
Adjust `--batch-size` for the available GPU memory, and use `--device` to select another GPU.

```text
calibration_data/
├── vision/
│   └── npy_files.json
├── prefill/
│   └── npy_files.json
├── decode/
│   └── npy_files.json
└── language/
    └── npy_files.json
```

Each vision sample has three inputs: folded pixel values `[1, 1, N, 1536]`, `pos_embeds` `[1, 1, N, 1024]`, and packed RoPE `[1, 1, N, 128]`.
The `npy_files.json` manifest marks the N axis dynamic.
Each decoder sample contains `inputs_embeds.npy`, three separate DeepStack files (`deepstack_0.npy`, `deepstack_1.npy`, and `deepstack_2.npy`), and a `cos.npy` RoPE tensor.
The embedding and DeepStack files have shape `[1, 1, T, 2048]`; `cos.npy` has shape `[1, 1, T, 256]` and feeds the decoder's runtime RoPE input.

The dataset revision, random seed, image order, and prompt order are fixed.
Repeated runs with the same options, GPU, and software environment produce identical calibration files.
Only generations that reach EOS are included.
If `./calibration_data` already exists, pass `--force` to replace it.

## 3. Compile MXQ Models

Compile the decoder first.
Decoder compilation produces the SpinR1 matrix required by encoder compilation and runtime model preparation.

For ARIES:

```bash
python compile_decoder.py --target-device aries-rb
python compile_encoder.py --target-device aries-rb
```

For REGULUS:

```bash
python compile_decoder.py --target-device regulus-rb
python compile_encoder.py --target-device regulus-rb
```

Each script loads the model with `load_for_part`, captures the inputs of one forward pass with `capture_forward_inputs`, creates its target-specific MBLT with `mblt_compile(model_part=...)`, and then compiles the MXQ model.
The decoder uses the `language` part.
The encoder uses the `vision` part with `model_part_options={"side_inputs": True}`, which makes the position embeddings, cosine, and sine graph inputs next to the folded pixel values, and marks their patch-count axis dynamic.
Compilation packs cosine and sine into one RoPE input, producing a three-input encoder MXQ.
Compiler options for both scripts are defined in `compile_config.py`.

```text
mblt/<target-device>/Qwen_Qwen3-VL-2B-Instruct_{decoder,encoder}.mblt
mxq/<target-device>/Qwen3-VL-2B-Instruct_{decoder,encoder}.mxq
spinWeight/<target-device>/Qwen3-VL-2B-Instruct/global_rotation.pth
```

The SpinR1 matrix path is scoped by `(target-device, model-name)` so multiple `--model-id` targets compiled against the same device do not overwrite each other.

The Qwen3-VL 2B compiler configuration is applied automatically.
ARIES uses `inference_scheme="all"`.
REGULUS uses `inference_scheme="single"` with a maximum sequence and cache length of 4096.

qbcompiler 1.4 enables bias correction by default (`BiasCorrectionConfig`), and `compile_config.py` keeps that default.
MXQ results and compile time therefore differ from builds made with qbcompiler 1.3.
GPU memory used during weight quantization is limited automatically (`ResourceManagementConfig.gpu_memory_budget_mb`, default `-1`); set a budget in MiB if the compile shares the GPU with other work.

## 4. Prepare the Runtime Model

Run this after both MXQ files have been compiled.

For ARIES:

```bash
python prepare_model.py --target-device aries-rb
```

For REGULUS:

```bash
python prepare_model.py --target-device regulus-rb
```

The script downloads the Mobilint runtime files, applies the decoder SpinR1 matrix to the token embedding, bundles `visual.pos_embed.weight` for the host-side position embeddings, copies both MXQ files, and writes the matching runtime configuration with top-level `dynamic_vision=true`.

The output is written to `./prepared/<target-device>/Qwen3-VL-2B-Instruct`.
If that directory already exists, pass `--force` to replace it.

## Output Layout

After compiling and preparing the ARIES build, the generated files are laid out as follows:

```text
images/

calibration_data/
├── vision/
├── prefill/
├── decode/
└── language/

mblt/aries-rb/
├── Qwen_Qwen3-VL-2B-Instruct_decoder.mblt
└── Qwen_Qwen3-VL-2B-Instruct_encoder.mblt

mxq/aries-rb/
├── Qwen3-VL-2B-Instruct_decoder.mxq
└── Qwen3-VL-2B-Instruct_encoder.mxq

spinWeight/aries-rb/
└── Qwen3-VL-2B-Instruct/

prepared/aries-rb/
└── Qwen3-VL-2B-Instruct/
```

## Other Model Sizes

Every compile / calibration / prepare script accepts `--model-id`.
The default is `Qwen/Qwen3-VL-2B-Instruct`; passing another id like `Qwen/Qwen3-VL-4B-Instruct` or `Qwen/Qwen3-VL-8B-Instruct` runs the same pipeline against that base model.
The runtime template repo id is derived as `mobilint/<name>` and Mobilint publishes `mobilint/Qwen3-VL-{2B,4B,8B}-Instruct`.

The compiler configuration in `compile_config.py` is configured for 2B.
Other model sizes require separate compilation and inference validation.
The 16-bit activation layers (decoder graph inputs, encoder graph outputs) are read from the MBLT by `compile_config.py`, so they follow the model size automatically.

## Runtime

Continue with the [Python VLM runtime tutorial](../../runtime/python/vlm/README.md).
Its default `--model-folder` points at the 2B ARIES prepared folder; for another target device or a non-2B `--model-id`, pass the matching folder explicitly:

```bash
python ../../runtime/python/vlm/inference_mblt_model_zoo.py \
    --model-folder prepared/aries-rb/Qwen3-VL-4B-Instruct
```
