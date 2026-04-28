# Calibration Data Guide

**Calibration data** is required during quantization to understand the distribution of model activations.
This document describes the purpose, format, and per-model preparation of calibration data.

## What is Calibration Data?

Quantization converts model float32 weights and activations to lower bits (8bit, 4bit).
To determine appropriate scales/offsets, the value ranges must be known accurately.

Calibration data consists of **representative input samples similar to actual inference data**.
The compiler passes these through the model to collect activation distributions at each layer.

Labels are not required. It is recommended to use data with a distribution similar to the input data during actual inference.

> **Note**: The shape of calibration data must match the input shape of the MBLT.

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
# 1. Load embedding weights
embedding_weight = torch.load("embedding.pt")
embedding_layer = torch.nn.Embedding.from_pretrained(embedding_weight)

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

```text
calibration_data/
├── language/
│   ├── sample_000/
│   │   └── inputs_embeds.npy
│   ├── npy_files.txt             # .npy file path list
│   └── metadata.json
└── vision/
    ├── sample_000/
    │   └── images.npy
    ├── npy_files.txt
    └── metadata.json
```

> **Tutorial reference**: `vlm/` (vision/language split), `stt/` (encoder/decoder split)

---

## `npy_files.txt` Format

A text file that lists calibration data paths, used in VLM, STT, etc.

```text
calibration_data/language/sample_000/inputs_embeds.npy
calibration_data/language/sample_001/inputs_embeds.npy
calibration_data/language/sample_002/inputs_embeds.npy
...
```

Pass this file path to `calib_data_path` of `mxq_compile()`:

```python
mxq_compile(
    ...,
    calib_data_path="calibration_data/language/npy_files.txt",
)
```

---

## Calibration Data Quality Tips

| Item | Recommendation |
|------|---------------|
| **Sample count** | Typically 100-128. Too few leads to biased distributions; too many increases compile time |
| **Diversity** | Use diverse samples with distributions similar to actual inference data |
| **Sequence length** | For LLMs, a minimum sequence length of 512 or more is recommended |
| **OOM handling** | Reduce calibration sample count or sequence length to resolve memory issues |

## Related Documents

- [Compile Config Guide](./01_about_quantization_config.md) - Quantization config that utilizes calibration data
- [Multi-Component Model Guide](./03_about_multi_component.md) - Split calibration structure for VLM/STT
