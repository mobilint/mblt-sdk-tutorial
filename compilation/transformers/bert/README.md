# Bidirectional Encoder Representations from Transformers

This tutorial provides detailed instructions for compiling a BERT model using the Mobilint Qubee compiler. The compilation process converts a standard BERT model into an optimized `.mxq` format that can run efficiently on Mobilint NPU hardware.

In this tutorial, we will use the [Sentence Bert](https://huggingface.co/sentence-transformers/msmarco-bert-base-dot-v5/tree/main) model, which is based on BERT architecture and modified to get sentence embeddings.

## Prerequisites

Before starting, ensure you have the following installed:

- Qubee SDK compiler installed (version >= 1.0.0 required)
- GPU with CUDA support (recommended for reducing compilation time)

Also, you need to install the following packages:

```bash
pip install accelerate datasets
```

