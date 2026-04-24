# Compilation Pipeline Overview

This document describes the overall flow of the model compilation pipeline using Mobilint qbcompiler.

## Overview

Models from PyTorch, TensorFlow, ONNX, etc. are designed to run inference on GPU/CPU.
To run these models on Mobilint NPU, they must be converted (compiled) into a format the NPU can understand.

qbcompiler supports model conversion from various frameworks.

The format of the original model is specified via the `backend` parameter:

| backend | Input Format | Example Model |
|---------|-------------|---------------|
| `"onnx"` | ONNX file | image_classification (`resnet50.onnx`) |
| `"torch"` | PyTorch model or HuggingFace model ID | llm (`meta-llama/Llama-3.2-1B-Instruct`) |
| `"hf"` | HuggingFace model (sub-model selectable) | stt (`openai/whisper-small` encoder/decoder) |

```python
from qbcompiler import mxq_compile, mblt_compile

# Converting an ONNX model (image_classification)
mxq_compile(model="./resnet50.onnx", backend="onnx", ...)

# Converting a PyTorch / HuggingFace model (llm)
mxq_compile(model="meta-llama/Llama-3.2-1B-Instruct", backend="torch", ...)

# Specifying a sub-model of a HuggingFace model (stt)
# model= takes a model object loaded via from_pretrained()
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small")
mblt_compile(model=model, backend="hf", target="encoder", ...)
mblt_compile(model=model, backend="hf", target="decoder", ...)
```

## Compilation Pipeline

The compilation process internally goes through two stages: **MBLT → MXQ** within qbcompiler.

![Compilation Pipeline](../../assets/compilation_pipeline.png)

### MBLT (Mobilint Binary LayouT)

A file that converts the original model's computation graph and weights into a hardware-agnostic intermediate format.

### MXQ (Mobilint eXeQutable)

The final deployment format that quantizes the MBLT and optimizes it for NPU hardware.
A `.mxq` file that can be directly executed on Mobilint NPU is generated.

---

## Compilation Methods

### One-step conversion with `mxq_compile()`

In most cases, passing the original model to `mxq_compile()` will
**automatically handle the MBLT → MXQ conversion internally**.

Users do not need to be aware of the intermediate MBLT stage.

```python
from qbcompiler import mxq_compile

mxq_compile(
    model="./resnet50.onnx",          # Original model path
    calib_data_path="./calib_data",   # Calibration data path
    save_path="./resnet50.mxq",       # MXQ save path
    backend="onnx",                   # Original model format
    device="gpu",                     # Compilation device ("gpu" or "cpu")
    inference_scheme="all",           # Inference scheme. "all" supports single, multi, global4, global8
)
```

### Explicitly generating MBLT

When a model consists of multiple sub-models (encoder/decoder, vision/language) like VLM or STT,
each component needs to be compiled individually since their inference call counts and quantization settings differ.

For example, in STT the encoder is called once while the decoder is called repeatedly for each token.
In VLM, the vision encoder is called once per image while the language model is called repeatedly for each token.

There are three ways to generate MBLT separately.

#### Method 1: Saving MBLT from `mxq_compile()`

By specifying `save_subgraph_type` and `output_subgraph_path` when calling `mxq_compile()`,
the MBLT file is saved alongside the MXQ conversion.

```python
mxq_compile(
    model="./resnet50.onnx",
    save_subgraph_type=2,                     # Also save MBLT file
    output_subgraph_path="./resnet50.mblt",   # MBLT save path
    ...
)
```

#### Method 2: Separate generation with `mblt_compile()`

When sub-models need to be compiled separately,
generate MBLT first with `mblt_compile()` then pass the MBLT path to `mxq_compile()`.

```python
from qbcompiler import mblt_compile, mxq_compile

# 1. Generate MBLT
mblt_compile(
    model=whisper_model,
    mblt_save_path="./whisper_encoder.mblt",
    backend="hf",
    target="encoder",
    device="cpu",
)

# 2. MBLT → MXQ conversion
mxq_compile(
    model="./whisper_encoder.mblt",   # Pass MBLT path
    calib_data_path="./calib_data",
    save_path="./whisper_encoder.mxq",
    ...
)
```

#### Method 3: Direct generation with `ModelParser`

When architectural patches (RoPE caching, convolution conversion, etc.) need to be applied before generating MBLT,
`ModelParser` is used to parse and serialize the model graph.

This is a low-level API, so rather than using it directly,
refer to the high-level functions provided in each tutorial.

> Example: `compile_language_model()` in `vlm/mblt_compile_language.py`,
> `compile_vision_encoder()` in `vlm/mblt_compile_vision.py`

> For split compilation of multi-component models, see
> [Multi-Component Model Guide](./03_about_multi_component.md).

---

## Per-Model Compilation Path Summary

| Model Type | backend | MBLT Generation | Notes |
|-----------|---------|----------------|-------|
| Image Classification | `onnx` | Automatic (inside mxq_compile) | |
| LLM | `torch` | Automatic (inside mxq_compile) | SpinQuant added for 4bit |
| BERT | `torch` | Automatic (inside mxq_compile) | |
| STT (Whisper) | `hf` | Explicit (mblt_compile) | encoder/decoder split |
| VLM (Qwen2-VL) | `torch` | Explicit (ModelParser) | vision/language split |

## Related Documents

- [Compile Config Guide](./01_about_quantization_config.md) - Detailed config options passed to `mxq_compile()`
- [Calibration Data Guide](./02_about_calibration_data.md) - Preparation and format of calibration data
- [Multi-Component Model Guide](./03_about_multi_component.md) - Models requiring split compilation (VLM/STT)
