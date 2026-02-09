# Large Language Model Inference

This tutorial provides step-by-step instructions for running inference with compiled large language models (LLMs) using the Mobilint qbruntime.

This guide is a continuation of [mblt-sdk-tutorial/compilation/transformers/llm/README.md](file:///workspace/mblt-sdk-tutorial/compilation/transformers/llm/README.md). It is assumed that you have successfully compiled the model and have the following files ready:

- `./Llama-3.2-1B-Instruct.mxq` - Compiled model file
- `./embedding.pt` - Embedding layer weights (PyTorch format)

## Prerequisites

Before running inference, ensure you have the following components installed and available:

- `qbruntime` library (to access the NPU accelerator)
- Compiled `.mxq` model file
- Embedding weights file (`.pt`)
- Python packages: `torch`, `transformers`

## Overview

The inference process uses a custom `LlamaMXQ` model class that integrates the NPU accelerator with the Hugging Face ecosystem. The workflow is as follows:

1.  **Initialization**: Load the compiled `.mxq` model onto the Mobilint NPU via the `qbruntime` accelerator.
2.  **Embedding**: Use a CPU-based embedding layer to convert input tokens into vectors.
3.  **Inference**: Process prompts through the NPU-accelerated transformer layers.
4.  **Generation**: Generate text using standard Hugging Face generation utilities.

## Running Inference

To run the example inference script, use the following command:

```bash
python inference_mxq.py --mxq-path ../../../compilation/transformers/llm/Llama-3.2-1B-Instruct.mxq --embedding-weight-path ../../../compilation/transformers/llm/embedding.pt
```

### Script Breakdown

- **Tokenizer Loading**: Loads the tokenizer from Hugging Face to process text input.
- **Model Initialization**: Initializes the `LlamaMXQ` model with the compiled `.mxq` file.
- **Generation**: Generates a response using the NPU-accelerated model.
- **Output**: Displays the generated text output.

### Parameters

- `--mxq-path`: Path to the compiled `.mxq` model file.
- `--embedding-weight-path`: Path to the embedding weights file (`.pt`).
- **Note**: The device is explicitly set to `'cpu'` in the script because the model offloads heavy computations to the NPU internally. Do **not** use GPU.

### Expected Output

The script will print the generated text response based on the input prompt.
