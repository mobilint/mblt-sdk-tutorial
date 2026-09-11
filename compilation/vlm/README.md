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

## Vision Modes

The pipeline can produce two flavors of the encoder:

- **Static (default)** — the vision graph bakes 224x224 into its position embeddings and rope tables, so every image the runtime sees is first resized to 224x224. Every step below defaults to this mode.
- **Dynamic** — position embeddings and the rope table become graph inputs so the runtime feeds the encoder images at their original resolution. Compiled artifacts get a `_dynamic` suffix so both variants can live side by side under `mblt/`, `mxq/`, and `prepared/`.

Dynamic vision is a *bundled release*: the compiled text decoder must also expose a runtime rope input (`--dynamic` on `compile_decoder.py`). `mblt-model-zoo` refuses to load a mismatched pair. If you want the dynamic path, pass `--dynamic` to every step below.

## 1. Download Calibration Images

```bash
python download_images.py
```

The script downloads 300 COCO validation images from a fixed dataset revision, converts them to RGB, and resizes them to `224x224` under `./images`.

For dynamic vision, skip the resize so the sample set spans a range of patch counts N:

```bash
python download_images.py --dynamic
```

## 2. Generate Calibration Data

```bash
python generate_calibration_data.py --batch-size 4
```

The script creates vision encoder samples and decoder prefill/decode samples under `./calibration_data`. The default batch size is 4 on `cuda:0`. Adjust `--batch-size` for the available GPU memory, and use `--device` to select another GPU.

```text
calibration_data/
├── vision/
│   └── npy_files.txt
├── prefill/
│   └── npy_files.json
├── decode/
│   └── npy_files.json
└── language/
    └── npy_files.json
```

Each vision sample contains `images.npy` with shape `[1024, 64, 6]`. Each decoder sample contains `inputs_embeds.npy` with shape `[1, T, 2048]` and one packed `deepstack_visual_embeds.npy` with shape `[3, T, 2048]`.

For dynamic vision, add `--dynamic` to switch every sample writer:

```bash
python generate_calibration_data.py --batch-size 4 --dynamic
```

Vision samples become 3-input (folded pixel values `[1, 1, N, 1536]`, `pos_embeds` `[1, 1, N, 1024]`, packed rope `[1, 1, N, 128]`) with `npy_files.json` marking the N axis dynamic. Decoder samples add a `cos.npy` `[1, T, 256]` rope tensor as third input for the runtime rope slot.

The dataset revision, random seed, image order, and prompt order are fixed. Repeated runs with the same options, GPU, and software environment produce identical calibration files. Only generations that reach EOS are included. If `./calibration_data` already exists, pass `--force` to replace it.

## 3. Compile MXQ Models

Compile the decoder first. Decoder compilation produces the SpinR1 matrix required by encoder compilation and runtime model preparation.

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

Each script creates its target-specific MBLT and then compiles the MXQ model. Compiler options for both scripts are defined in `compile_config.py`.

```text
mblt/<target-device>/Qwen_Qwen3-VL-2B-Instruct_{decoder,encoder}.mblt
mxq/<target-device>/Qwen3-VL-2B-Instruct_{decoder,encoder}.mxq
spinWeight/<target-device>/Qwen3-VL-2B-Instruct/global_rotation.pth
```

Adding `--dynamic` produces `_dynamic`-suffixed peers alongside the static artifacts (e.g. `Qwen3-VL-2B-Instruct_{decoder,encoder}_dynamic.mxq` and `spinWeight/<target-device>/Qwen3-VL-2B-Instruct-dynamic/global_rotation.pth`). The SpinR1 matrix path is scoped by `(target-device, model-name, mode)` so multiple `--model-id` targets compiled against the same device do not overwrite each other.

The validated Qwen3-VL 2B compiler configuration is applied automatically. ARIES uses `inference_scheme="all"`. REGULUS uses `inference_scheme="single"` with a maximum sequence and cache length of 1024.

For dynamic vision, pass `--dynamic` to both scripts (decoder first, again):

```bash
python compile_decoder.py --target-device aries-rb --dynamic
python compile_encoder.py --target-device aries-rb --dynamic
```

The decoder promotes its cos/sin `InputConstant`s to a runtime rope input via `LlmConfig.attributes.runtime.dynamic_rope=True`. The encoder switches to the V2 dispatch (`qbcompiler.model_dict`) that treats `pos_embeds`, `cos`, and `sin` as graph inputs and marks the N axis dynamic. Compiled artifacts land at `mxq/<target-device>/Qwen3-VL-2B-Instruct_{decoder,encoder}_dynamic.mxq`.

`ENCODER_16BIT_ACTIVATIONS` in `compile_config.py` lists graph-derived operator names measured on the static Qwen3-VL-2B build; dynamic parsing (or a different model size) can emit different names. Unmatched entries are silently ignored by the quantizer, so a wrong list costs a small amount of vision SQNR but does not break the build.

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

The script downloads the Mobilint runtime files, applies the decoder SpinR1 matrix to the token embedding, copies both MXQ files, and writes the matching runtime configuration.

The output is written to `./prepared/<target-device>/Qwen3-VL-2B-Instruct`. If that directory already exists, pass `--force` to replace it.

For dynamic vision:

```bash
python prepare_model.py --target-device aries-rb --dynamic
```

This picks up the `_dynamic` MXQ pair, additionally bundles `visual.pos_embed.weight` into `model.safetensors` (only the dynamic runtime path allocates that submodule), sets top-level `dynamic_vision=true` in `config.json`, and writes to `./prepared/<target-device>/Qwen3-VL-2B-Instruct-dynamic`.

## Other Model Sizes

Every compile / calibration / prepare script accepts `--model-id`. The default is `Qwen/Qwen3-VL-2B-Instruct`; passing another id like `Qwen/Qwen3-VL-4B-Instruct` or `Qwen/Qwen3-VL-8B-Instruct` runs the same pipeline against that base model. The runtime template repo id is derived as `mobilint/<name>` and Mobilint publishes `mobilint/Qwen3-VL-{2B,4B,8B}-Instruct`.

The compiler configuration in `compile_config.py` is tuned for 2B. Compilation succeeds on other sizes but the `ENCODER_16BIT_ACTIVATIONS` list may not match the new graph, and the ARIES branch's `hessian_quant_config=None` was validated only against 2B. Both are quality knobs, not correctness constraints.

## Runtime

Continue with the [Python VLM runtime tutorial](../../runtime/python/vlm/README.md). Its default `--model-folder` points at the static 2B prepared folder; for a dynamic build or a non-2B `--model-id`, pass the matching folder explicitly:

```bash
# Dynamic 2B
python ../../runtime/python/vlm/inference_mblt_model_zoo.py \
    --model-folder prepared/aries-rb/Qwen3-VL-2B-Instruct-dynamic

# Static 4B (or 8B): drop the `-dynamic` suffix
python ../../runtime/python/vlm/inference_mblt_model_zoo.py \
    --model-folder prepared/aries-rb/Qwen3-VL-4B-Instruct
```
