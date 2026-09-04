# LLM Runtime

Complete the [LLM compilation tutorial](../../../compilation/llm/README.md) first.

## Prerequisites

```bash
pip install -r requirements.txt
```

## `mblt-model-zoo` Inference

Use a model folder created by `compilation/llm/prepare_models.py`.

```bash
python inference_mblt_model_zoo.py \
  --model-folder ../../../compilation/llm/llama-mxq-w8
```

Useful options:

```bash
python inference_mblt_model_zoo.py --prompt "What is quantum computing?"
python inference_mblt_model_zoo.py --max-new-tokens 512
```

## Direct `qbruntime` Inference

The direct wrapper reads the MXQ and the prepared `model.safetensors` file.

```bash
python inference_mxq.py \
  --mxq-path ../../../compilation/llm/Llama-3.2-1B-Instruct-W8.mxq \
  --embedding-path ../../../compilation/llm/llama-mxq-w8/model.safetensors
```

Both scripts accept `--prompt` and `--max-new-tokens`.
