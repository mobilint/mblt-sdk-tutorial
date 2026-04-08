# Model Preparation Guide

To use the `mblt-model-zoo` API, compilation outputs must be organized into a model folder.
This document describes what `prepare_model.py` does and the structure of `config.json`.

> Simple models (Image Classification, Object Detection, etc.)
> pass `.mxq` files directly to `qbruntime`, so this preparation step is not needed.

---

## What prepare_model.py Does

The `prepare_model.py` provided in each tutorial directory
converts compilation outputs into a folder structure recognized by `mblt-model-zoo`.

Tasks performed:

1. **Copy `.mxq` files** — Copy compilation outputs to the output folder
2. **Convert embedding weights** — Convert `.pt` to safetensors format (model-dependent)
3. **Configure config.json** — Add NPU settings such as `mxq_path`, `_name_or_path`, `target_cores`
4. **Download tokenizer** — Download tokenizer files from HuggingFace (for text models)

**Usage examples**:
- `llm/prepare_model.py`
- `vlm/prepare_model.py`
- `stt/prepare_model.py`

---

## config.json Structure

`config.json` is based on the HuggingFace standard config with NPU-specific fields added.

### Common Fields

```json
{
    // HuggingFace standard fields
    "architectures": ["MobilintLlamaForCausalLM"],
    "model_type": "mobilint-llama",

    // ID for mblt-model-zoo to auto-discover the model
    "_name_or_path": "mobilint/Llama-3.2-1B-Instruct",

    // Compiled MXQ file path (relative to model folder)
    "mxq_path": "Llama-3.2-1B-Instruct.mxq",

    // NPU core allocation
    "target_cores": ["0:0"]
}
```

### Multi-Component Models

When there are multiple sub-models, each sub-model's `mxq_path` and `target_cores` are specified separately.

**VLM** (vision + language):

```json
{
    "mxq_path": "text_model.mxq",
    "target_cores": ["0:0"],
    "vision_config": {
        "mxq_path": "vision_transformer.mxq",
        "target_cores": ["0:0"]
    }
}
```

**STT** (encoder + decoder):

```json
{
    "encoder_mxq_path": "whisper_encoder.mxq",
    "encoder_target_cores": ["0:0"],
    "decoder_mxq_path": "whisper_decoder.mxq",
    "decoder_target_cores": ["0:0"]
}
```

---

## NPU Core Allocation

`target_cores` specifies which NPU core(s) the model runs on.

> For detailed explanation of core modes, see
> [Mobilint Multi-Core Documentation](https://docs.mobilint.com/v1.0/en/multicore.html).

### Core Modes

| Mode | Setting | Description |
|------|---------|-------------|
| single | `"target_cores": ["0:0"]` | Run on a single core |
| multi | `"core_mode": "multi", "target_clusters": [0]` | Multiple cores collaborate within one cluster |
| global4 | `"core_mode": "global4", "target_clusters": [0]` | Use 4 cores in one cluster |
| global8 | `"core_mode": "global8", "target_clusters": [0, 1]` | Use 8 cores across two clusters |

### target_cores Format

Specified in `"cluster:core"` format.

```text
"0:0" → Cluster 0, Core 0
"0:1" → Cluster 0, Core 1
"1:0" → Cluster 1, Core 0
```

### When Using multi/global Modes

Use `core_mode` and `target_clusters` instead of `target_cores`.

```json
{
    "core_mode": "global4",
    "target_clusters": [0]
}
```

> If compiled with `inference_scheme="all"`, all core modes can be used.
> If compiled with a specific scheme, only that mode is available.

---

## Next Documents

- [Runtime Pipeline Overview](./00_about_runtime_pipeline.md) - Overall flow
- [Inference API Guide](./02_about_inference_api.md) - qbruntime vs mblt-model-zoo
