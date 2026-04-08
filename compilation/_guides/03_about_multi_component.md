# Multi-Component Model Guide

Models composed of multiple sub-models, such as VLM (Vision-Language Model) or STT (Speech-to-Text),
must have each component **compiled individually**.

This document describes when split compilation is needed and the key considerations.

---

## Why is Split Compilation Needed?

Single models (e.g., ResNet, BERT) are compiled into a single `.mxq` file.

However, models composed of multiple sub-models require splitting due to architectural characteristics:

- **Different input formats between sub-models** — e.g., in VLM, the vision encoder takes images while the language model takes text embeddings
- **Different inference call counts between sub-models** — e.g., in STT, the encoder is called once while the decoder is called repeatedly for each token
- **Different quantization settings per sub-model** — e.g., an autoregressive decoder requires `LlmConfig` but the encoder does not

---

## Compilation Order Dependencies

There may be compilation order dependencies between sub-models.

### With Order Dependency (VLM)

When using SpinQuant's R1,
the R1 rotation matrix generated during language model compilation must be referenced during vision encoder compilation.

```text
[1] Language Model compilation  ──→  R1 rotation matrix generated (spinWeight/)
                                              │
[2] Vision Encoder compilation  ←──  R1 referenced (HeadOutChRotation)
```

Since the vision encoder's output must align with the rotated language model's input space,
**the language model must be compiled first**.

> See the `vlm/` tutorial for detailed implementation.

### Without Order Dependency (STT)

When weight space alignment between encoder and decoder is not needed,
each component can be compiled independently.

```text
[1] Encoder compilation  (independent)
[2] Decoder compilation  (independent)
```

Note that if the decoder has an autoregressive structure, `LlmConfig` must be passed.
The encoder receives fixed-length input, so `LlmConfig` is not needed.

> See the `stt/` tutorial for detailed implementation.

---

## Embedding Layer Separation

In VLM, LLM, BERT, etc., **the embedding layer does not run on NPU**.

Therefore:

1. Extract embedding weights separately
2. Perform embedding lookup on CPU
3. Feed the result into the NPU model

When using SpinQuant(R1), the extracted embeddings must have R1 rotation applied.

> For details on embedding rotation, see
> [Compile Config Guide - SpinQuant Details](./01_about_quantization_config.md#spinquant-r1r2-details).

---

## Final Deployment Package

The compilation output of multi-component models consists of multiple `.mxq` files and configuration files.

```text
mxq/
├── {component_1}.mxq          # Sub-model 1
├── {component_2}.mxq          # Sub-model 2
├── config.json                # Configuration file with mxq_path added
└── model.safetensors          # Embedding weights (with rotation applied if needed)
```

`config.json` includes the `.mxq` path (`mxq_path`) for each sub-model,
enabling the runtime to load the correct model files.

---

## Next Documents

- [Compilation Pipeline Overview](./00_about_compilation_pipeline.md) - Overall compilation flow
- [Compile Config Guide](./01_about_quantization_config.md) - Per-component quantization settings
- [Calibration Data Guide](./02_about_calibration_data.md) - Per-component calibration data structure
