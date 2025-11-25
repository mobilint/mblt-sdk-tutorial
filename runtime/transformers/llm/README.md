# Large Language Model Runtime Guide

This tutorial provides detailed instructions for running inference with compiled large language models using the Mobilint qb runtime.

This guide continues from `mblt-sdk-tutorial/compilation/transformers/llm/README.md`. We assume you have successfully compiled the model and have the following files ready:

- `./Llama-3.2-1B-Instruct.mxq` - Compiled model file
- `./embedding.pt` - Embedding layer weights

## Prerequisites

Before running inference, ensure you have:

- maccel runtime library (provides NPU accelerator access)
- Compiled `.mxq` model file
- Embedding weights file

## Overview

The inference process uses a custom `LlamaMXQ` model class that:

1. Loads the compiled `.mxq` model onto the Mobilint NPU via maccel accelerator
2. Uses CPU-based embedding layer for token-to-vector conversion
3. Processes prompts through the NPU-accelerated transformer layers
4. Generates text using standard HuggingFace generation utilities

---

## Running Inference

To run the example inference script:

```bash
python inference_mxq.py --mxq_path ../../../compilation/transformers/llm/Llama-3.2-1B-Instruct.mxq --embedding_weight_path ../../../compilation/transformers/llm/embedding.pt
```

**What this does:**

- Loads the tokenizer from HuggingFace
- Initializes the `LlamaMXQ` model with the compiled `.mxq` file
- Generates a response using the NPU-accelerated model
- Displays the generated text output

**Important Configuration:**

- Set the `MODEL_NAME` to an appropriate model id
- Device is set to `'cpu'` - Do NOT use GPU as the model runs on NPU
