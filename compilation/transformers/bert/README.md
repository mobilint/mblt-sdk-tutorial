# Bidirectional Encoder Representations from Transformers

This tutorial provides detailed instructions for compiling a BERT model using the Mobilint Qubee compiler. The compilation process converts a standard BERT model into an optimized `.mxq` format that can run efficiently on Mobilint NPU hardware.

In this tutorial, we will use the [Korean Sentence Bert (STS finetuned)](https://huggingface.co/jhgan/ko-sbert-sts).

## Prerequisites

Before starting, ensure you have the following installed:

- Qubee SDK compiler installed (version >= 0.12 required)
- GPU with CUDA support (recommended for reducing compilation time)
- HuggingFace account with access to Llama models (if using gated models)

Also, you need to install the following packages:

```bash
pip install accelerate datasets
```

## Model Analysis

Due to the complex architecture, some layers of BERT may not be supported by the compiler. Therefore, model analysis should be performed prior to the compilation process.

To analyze the model structure, we convert the model to MBLT format.

```python
from qubee import mblt_compile
from qubee.model_dict.parser.backend.torch.object_wrapper import set_attention_mask
from qubee.model_dict.parser.backend.torch.util import wrap_tensor
from transformers import BertModel, BertTokenizer

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(
        "jhgan/ko-sbert-sts",
        trust_remote_code=True,
    )
    model = BertModel.from_pretrained("jhgan/ko-sbert-sts", trust_remote_code=True)
    model.eval()

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    feed_dict = {}
    for k, v in inputs.items():
        wrapped = wrap_tensor(k, v)
        wrapped.src_shape[1].set_dynamic()
        feed_dict[k] = wrapped
    set_attention_mask(feed_dict["attention_mask"], "padding_mask")

    mblt_compile(
        model=model,
        mblt_save_path="ko-sbert-sts.mblt",
        backend="torch",
        feed_dict=feed_dict,
        cpu_offload=True,
    )

```

After running `mblt_compile.py`, you will get a `ko-sbert-sts.mblt` file. You can visualize the architecture of the model using [Netron](https://netron.mobilint.com).

According to the Netron visualization, some input embedding related layers are not supported. Therefore, we need to prepare a calibration dataset to match the input of the supported layers, and execute unsupported operations outside of the Mobilint NPU.

## Calibration Dataset Preparation

### Get Embedding Weights

As displayed on Netron, the input embedding part is not supported. Therefore, we need to prepare the calibration dataset to match the input of the supported layers. To achieve this, we extract the embedding weights from the model and save them as a `.pth` file.

This can be done by running the code `get_embedding.py`.

```python
import torch
from transformers import BertModel

model = BertModel.from_pretrained("jhgan/ko-sbert-sts", trust_remote_code=True)

word_embeddings = model.embeddings.word_embeddings.weight
token_type_embeddings = model.embeddings.token_type_embeddings.weight
position_embeddings = model.embeddings.position_embeddings.weight
layernorm_weight = model.embeddings.LayerNorm.weight
layernorm_bias = model.embeddings.LayerNorm.bias

print(word_embeddings.shape)
print(token_type_embeddings.shape)
print(position_embeddings.shape)
print(layernorm_weight.shape)
print(layernorm_bias.shape)
weight_dict = {
    "word_embeddings": word_embeddings,
    "token_type_embeddings": token_type_embeddings,
    "position_embeddings": position_embeddings,
    "layernorm_weight": layernorm_weight,
    "layernorm_bias": layernorm_bias,
}

torch.save(weight_dict, "weight_dict.pth")
```

After running the code, you will get a `weight_dict.pth` file.

### Prepare Calibration Dataset

We will generate the calibration dataset using the [Korean STS benchmark dataset](https://huggingface.co/datasets/mteb/KorSTS), which is managed by the Massive Text Embedding Benchmark (MTEB).

We first tokenize the sentences from the dataset and embed them using the pre-extracted embedding weights. Then, we save the embedded text as a numpy file.

This can be done by running `prepare_calib.py`. After running the code, you will get a `calib` directory.

## Model Compilation

With the calibration dataset, we can compile the model using the Mobilint Qubee compiler.

```python
from qubee import mxq_compile
from qubee.model_dict.parser.backend.torch.object_wrapper import set_attention_mask
from qubee.model_dict.parser.backend.torch.util import wrap_tensor
from transformers import BertModel, BertTokenizer

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(
        "jhgan/ko-sbert-sts",
        trust_remote_code=True,
    )
    model = BertModel.from_pretrained("jhgan/ko-sbert-sts", trust_remote_code=True)
    model.eval()

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    feed_dict = {}
    for k, v in inputs.items():
        wrapped = wrap_tensor(k, v)
        wrapped.src_shape[1].set_dynamic()
        feed_dict[k] = wrapped
    set_attention_mask(feed_dict["attention_mask"], "padding_mask")

    mxq_compile(
        model=model,
        save_path="ko-sbert-sts.mxq",
        calib_data_path="./calib",
        backend="torch",
        feed_dict=feed_dict,
    )
```

After running the code `compile_mxq.py`, you will get `ko-sbert-sts.mxq`.
