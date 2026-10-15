# Calibration Data Guide

**Calibration data** is required during quantization to understand the distribution of model activations.
This document describes the purpose, format, and per-model preparation of calibration data.

## What is Calibration Data?

Quantization converts model float32 weights and activations to lower bits (8bit, 4bit).
To determine appropriate scales/offsets, the value ranges must be known accurately.

Calibration data consists of **representative input samples similar to actual inference data**.
The compiler passes these through the model to collect activation distributions at each layer.

Labels are not required. It is recommended to use data with a distribution similar to the input data during actual inference.

> **Note**: The calibration data must match the inputs of the model that the quantizer builds from the MBLT. See the `npy_files.json` section below.

```text
calibration data → pass through model → collect activation distributions → determine quantization scales
```

> **Difference from PyTorch quantization**: PyTorch's `torch.quantization` uses calibration optionally,
> but in qbcompiler, calibration is mandatory for NPU optimization.

---

## Calibration Data Examples by Model Type

### Image Input Models

Simply collect raw image files in a directory.
When using `PreprocessingConfig`, the compiler automatically performs preprocessing, so raw images can be used as-is.

```text
calib_data/
├── image_001.JPEG
├── image_002.JPEG
└── ...
```

- **How to pass**: Provide the directory path to `calib_data_path`

> **Tutorial reference**: `image_classification/`

---

### Embedding Input Models (Single Input)

For models where the embedding layer does not run on NPU,
**`.npy` files of embedding vectors** converted from tokenized text are used as calibration data.

```text
calibration_data/
├── inputs_embeds_0.npy     # shape: [1, seq_len, embed_dim]
├── inputs_embeds_1.npy
└── ...
```

Generation process:

```python
# 1. Load the model's embedding layer
model = AutoModelForCausalLM.from_pretrained(model_id)
embedding_layer = model.get_input_embeddings()

# 2. Tokenize text
token_ids = tokenizer(text, return_tensors="pt")["input_ids"]

# 3. Convert token IDs → embedding vectors
embedded_text = embedding_layer(token_ids)  # [1, seq_len, embed_dim]

# 4. Save as .npy
np.save("inputs_embeds_0.npy", embedded_text.numpy())
```

> **Why embeddings are extracted separately**: The NPU does not support embedding lookup table operations,
> so the embedding layer runs on CPU and its output is fed into the NPU model.
> Therefore, calibration data must also be post-embedding values.
> **Tutorial reference**: `llm/`, `bert/`

---

### Multi-Input Models

For models with multiple input tensors, `.npy` file paths for each input must be mapped as pairs.
Since a simple `.npy` list cannot represent multi-input mapping, **JSON format** is used.

```text
calibration_data/
├── sample_0000/
│   ├── decoder_hidden_states.npy
│   └── encoder_hidden_states.npy
├── sample_0001/
│   └── ...
└── calib.json                    # Multi-input path mapping
```

> **Tutorial reference**: `stt/` (encoder/decoder split)

---

### Multi-Component Models (Per Sub-model Split)

Models composed of multiple sub-models require **separate calibration data for each sub-model**,
since each sub-model has a different input format.

The VLM tutorial writes one directory for each sub-model. The layout depends on the vision mode.

Dynamic mode (default). The vision samples have three inputs, and the language samples have five:

```text
calibration_data/
├── vision/
│   ├── sample_000/
│   │   ├── float_1_channel_last.npy   # folded pixel values [1, 1, N, 1536]
│   │   ├── pos_embeds.npy             # position embeddings [1, 1, N, 1024]
│   │   └── cos.npy                    # packed RoPE [1, 1, N, 128]
│   └── npy_files.json                 # input manifest, 3 inputs
├── prefill/                           # staging
│   ├── sample_000/
│   └── npy_files.json                 # not used by the compile
├── decode/                            # staging
│   ├── sample_000/
│   └── npy_files.json                 # not used by the compile
└── language/                          # prefill and decode merged; the decoder compile uses this directory
    ├── prefill_000/
    │   ├── inputs_embeds.npy          # [1, 1, T, 2048]
    │   ├── deepstack_0.npy            # [1, 1, T, 2048]
    │   ├── deepstack_1.npy
    │   ├── deepstack_2.npy
    │   └── cos.npy                    # packed RoPE [1, 1, T, 256]
    ├── decode_000/
    │   └── ...                        # the same five files
    └── npy_files.json                 # input manifest, 5 inputs
```

Static mode (`--static`). The vision samples have one input, so the vision list is a `.txt` file:

```text
calibration_data_static/
├── vision/
│   ├── sample_000/
│   │   └── images.npy                 # folded pixel values [1, 256, 1536]
│   └── npy_files.txt                  # path list, 1 input
├── prefill/                           # staging
│   ├── sample_000/
│   └── npy_files.json                 # not used by the compile
├── decode/                            # staging
│   ├── sample_000/
│   └── npy_files.json                 # not used by the compile
└── language/                          # prefill and decode merged; the decoder compile uses this directory
    ├── prefill_000/
    │   ├── inputs_embeds.npy
    │   ├── deepstack_0.npy
    │   ├── deepstack_1.npy
    │   └── deepstack_2.npy            # no cos.npy: dynamic_rope=False, so the quantizer adds no RoPE input
    ├── decode_000/
    │   └── ...
    └── npy_files.json                 # input manifest, 4 inputs
```

> **Tutorial reference**: `vlm/` (vision/language split), `stt/` (encoder/decoder split)

---

## `npy_files.txt` Format

A text file that lists calibration data paths.
Use it when the sub-model has one input, for example the static VLM vision mode (`npy_files.txt`) and the STT encoder (`whisper_encoder_cali.txt`).
The generated file uses absolute paths.

```text
calibration_data_static/vision/sample_000/images.npy
calibration_data_static/vision/sample_001/images.npy
calibration_data_static/vision/sample_002/images.npy
...
```

Pass this file path to `calib_data_path` of `mxq_compile()`:

```python
mxq_compile(
    ...,
    target_device="aries-rb",
    calib_data_path="calibration_data_static/vision/npy_files.txt",
)
```

---

## `npy_files.json` Format

A JSON manifest that lists the input names, the input shapes, and one path list for each sample.
Use it when the sub-model has several inputs, for example the Qwen3-VL vision and language parts.
The VLM tutorial writes this manifest in the default dynamic mode.
This is the dynamic vision manifest, shortened to one sample:

```json
{
    "info": {
        "input names": ["float_1_channel_last", "pos_embeds/reshape", "cos"],
        "input shapes": [[1, 1, -1, 1536], [1, 1, -1, 1024], [1, 1, -1, 128]]
    },
    "calib paths": [
        [
            "calibration_data/vision/sample_000/float_1_channel_last.npy",
            "calibration_data/vision/sample_000/pos_embeds.npy",
            "calibration_data/vision/sample_000/cos.npy"
        ]
    ]
}
```

The generated file uses absolute paths.
A `-1` axis marks a dynamic axis.
The input count and shapes must match the inputs of the model that the quantizer builds from the MBLT.
These inputs can differ from the MBLT graph inputs, for example:

- The Qwen3-VL vision MBLT has separate `cos` and `sin` inputs of width 64. The quantizer merges them into one RoPE input of width 128, so the manifest lists one `cos` entry of width 128.
- The Qwen3-VL decoder MBLT has four inputs and no RoPE input. In dynamic mode the compile config sets `dynamic_rope=True`, and the quantizer adds a RoPE input of width 256. The language manifest therefore lists five inputs: four inputs of width 2048 (`inputs_embeds/reshape`, `deepstack_visual_embeds/reshape/slice`, `deepstack_visual_embeds/reshape/slice_0`, `deepstack_visual_embeds/reshape/slice_1`) and `cos` of width 256. In static mode `dynamic_rope=False`, so the manifest lists the four inputs only.

When several inputs share one shape, the quantizer matches the files by input name and rejects a name that is not a model input name.
When a shape is unique, the quantizer does not check the name.
Keep the names equal to the model input names in every case.

---

## Calibration Data Quality Tips

| Item | Recommendation |
| ------ | --------------- |
| **Sample count** | Typically 100-128. Too few leads to biased distributions; too many increases compile time |
| **Diversity** | Use diverse samples with distributions similar to actual inference data |
| **Sequence length** | For LLMs, a minimum sequence length of 512 or more is recommended |
| **OOM handling** | Reduce calibration sample count or sequence length to resolve memory issues. Since qbcompiler 1.4, GPU memory for weight quantization is budgeted automatically; set `ResourceManagementConfig(gpu_memory_budget_mb=...)` (MiB, `-1` auto, `0` unlimited) to change it |

## Related Documents

- [Compile Config Guide](./01_about_quantization_config.md) - Quantization config that utilizes calibration data
- [Multi-Component Model Guide](./03_about_multi_component.md) - Split calibration structure for VLM/STT
