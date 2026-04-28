# Inference API Guide

This document introduces the two APIs for running inference on Mobilint NPU.

> This assumes compiled `.mxq` files are ready.
> For the compilation process, see [Compilation Pipeline Overview](../../compilation/_guides/00_about_compilation_pipeline.md).

---

## API Types

| API | Level | Features | Suitable Models |
|-----|-------|----------|----------------|
| `qbruntime` | low-level | Direct NPU control, numpy I/O | Simple models (Image Classification, Object Detection, BERT, etc.) |
| `mblt-model-zoo` | high-level | HuggingFace-compatible API, automatic NPU management | Complex models (LLM, VLM, STT, etc.) |

---

## qbruntime

A low-level API for direct NPU control.
Loads `.mxq` files and runs inference with numpy arrays as input.

```python
import numpy as np
import qbruntime

# 1. Create NPU accelerator
#    Argument is the device number (starting from 0)
acc = qbruntime.Accelerator(0)

# 2. Model configuration
mc = qbruntime.ModelConfig()

# Core allocation setting.
# Specify a particular core with CoreId(cluster, core).
mc.set_single_core_mode(
    None,
    [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)]
)

# 3. Load model and deploy to NPU
model = qbruntime.Model("./model.mxq", mc)
model.launch(acc)

# 4. Run inference
#    Input: numpy array (dtype and shape depend on the model)
#    Output: list of numpy arrays
output = model.infer(input_numpy_array)

# 5. Resource cleanup
model.dispose()
```

### Input Data Format

Models compiled with `PreprocessingConfig` to fuse preprocessing
accept **raw images in UInt8 format** as input.

```python
# Convert image to numpy uint8 array
image = np.array(pil_image, dtype=np.uint8)  # [H, W, C]
output = model.infer(image)
```

Separate normalization is not needed.

**Usage examples**:
- `image_classification/inference_mxq.py`
- `object_detection/inference_mxq.py`
- `pose_estimation/inference_mxq.py`

---

## mblt-model-zoo

A high-level API compatible with the HuggingFace `transformers` API.
Automatically loads models and manages NPU resources based on settings in the model folder's `config.json`,
including `mxq_path`, `target_cores`, `_name_or_path`, etc.

> For the config.json structure, see
> [Model Preparation Guide](./01_about_model_preparation.md).

```python
# Import mblt-model-zoo model implementation to register with HuggingFace
# Without this import, the architectures in config.json won't be recognized
import mblt_model_zoo.hf_transformers.models.llama.modeling_llama  # noqa: F401

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# 1. Load model
#    Automatically reads mxq_path, target_cores, etc. from config.json
model = AutoModelForCausalLM.from_pretrained("./model-folder")

# 2. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("model-id")

# 3. Run inference
#    Uses HuggingFace standard generate() API
inputs = tokenizer(["prompt text"], return_tensors="pt")
streamer = TextStreamer(tokenizer, skip_prompt=True)

output = model.generate(
    **inputs,
    max_new_tokens=256,
    streamer=streamer,      # Real-time output as tokens are generated
)

# 4. Resource cleanup
model.dispose()
```

### mblt-model-zoo Import Rules

For `from_pretrained()` to recognize the `architectures` field in config.json,
the corresponding model implementation must be imported beforehand.

```python
# LLM
import mblt_model_zoo.hf_transformers.models.llama.modeling_llama

# VLM
import mblt_model_zoo.hf_transformers.models.qwen2_vl.modeling_qwen2_vl

# STT
import mblt_model_zoo.hf_transformers.models.whisper.modeling_whisper
```

**Usage examples**:
- `llm/inference_mblt_model_zoo.py`
- `vlm/inference_mblt_model_zoo.py`
- `stt/inference_mblt_model_zoo.py`

---

## Split Execution (CPU/NPU Separation)

In complex models, not all layers run on the NPU.
The embedding layer runs on CPU since it is a lookup operation that retrieves vectors from a weight table using token IDs.

Which layers run on CPU vs NPU depends on the model and compilation configuration.

This separation is handled automatically by `mblt-model-zoo`.
When using `qbruntime` directly, it is implemented in wrapper classes.

> Wrapper class examples: `bert/wrapper/bert_model.py`, `llm/wrapper/llama_model.py`

---

## KV Cache Management

Models with transformer decoder architecture (LLM, STT decoder)
cache previous tokens' key/value during autoregressive generation.

KV cache is managed internally by the NPU, and must be initialized when starting a new generation.

```python
# When using mblt-model-zoo, dispose() also cleans up the cache
model.dispose()

# When using qbruntime wrapper, explicitly initialize the cache
model.mxq_model.dump_cache_memory()
```

---

## Related Documents

- [Runtime Pipeline Overview](./00_about_runtime_pipeline.md) - Overall flow
- [Model Preparation Guide](./01_about_model_preparation.md) - config.json structure and NPU core allocation
