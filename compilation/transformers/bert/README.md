# Bidirectional Encoder Representations from Transformers (BERT)

This tutorial provides detailed instructions for compiling a BERT model using the Mobilint qb Compiler. The compilation process converts a standard BERT model into an optimized `.mxq` format that can run efficiently on Mobilint NPU hardware.

In this tutorial, we will use the [Sentence-BERT](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) model, which is based on the BERT architecture and modified to generate sentence embeddings.

## Prerequisites

Before starting, ensure you have the following installed:

- Mobilint qb Compiler (version >= 1.0.0 required)
- GPU with CUDA support (recommended for reducing compilation time)

Additionally, you need to install the following packages:

```bash
pip install accelerate datasets
```

## Model Analysis

Due to its complex architecture, some layers of BERT may not be supported by the compiler. Therefore, model analysis should be performed before the compilation process.

To analyze the model structure, we convert the model to the MBLT format.

```python
from qbcompiler import mblt_compile
from qbcompiler.model_dict.parser.backend.torch.object_wrapper import set_attention_mask
from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor
from transformers import BertModel, BertTokenizer

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        trust_remote_code=True,
    )
    model = BertModel.from_pretrained("sentence-transformers-testing/stsb-bert-tiny-safetensors", trust_remote_code=True)
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
        mblt_save_path="stsb-bert-tiny-safetensors.mblt",
        backend="torch",
        feed_dict=feed_dict,
        cpu_offload=True,
    )
```

After running `compile_mblt.py`, you will obtain a `stsb-bert-tiny-safetensors.mblt` file. You can visualize the model architecture using [Netron](https://netron.mobilint.com).

Based on the Netron visualization, some input embedding layers are not supported. Therefore, we need to prepare a calibration dataset to match the input of the supported layers and execute unsupported operations outside of the Mobilint NPU.

## Calibration Dataset Preparation

### Extracting Embedding Weights

As shown in Netron, the input embedding section is not supported. Therefore, we need to prepare a calibration dataset to match the input of the supported layers. To achieve this, we extract the embedding weights from the model and save them as a `.pth` file.

This can be done by running `get_embedding.py`.

```python
import torch

from transformers import BertModel

model = BertModel.from_pretrained(
    "sentence-transformers-testing/stsb-bert-tiny-safetensors", trust_remote_code=True
)

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

After running the script, you will get a `weight_dict.pth` file.

### Preparing the Calibration Dataset

We will generate the calibration dataset using the [STS Benchmark Dataset](https://huggingface.co/datasets/mteb/stsbenchmark-sts), which is managed by the Massive Text Embedding Benchmark (MTEB).

First, we tokenize the sentences from the dataset and embed them using the previously extracted embedding weights. Then, we save the embedded text as NumPy files.

This can be done by running `prepare_calib.py`. This will create a `calib` directory.

## Model Compilation

With the calibration dataset ready, we can compile the model using the Mobilint qb Compiler.

```python
from qbcompiler import mxq_compile, QuantizationConfig
from qbcompiler.model_dict.parser.backend.torch.object_wrapper import set_attention_mask
from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor

from transformers import BertModel, BertTokenizer

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        trust_remote_code=True,
    )
    model = BertModel.from_pretrained(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors", trust_remote_code=True
    )
    model.eval()

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    feed_dict = {}
    for k, v in inputs.items():
        wrapped = wrap_tensor(k, v)
        wrapped.src_shape[1].set_dynamic()
        feed_dict[k] = wrapped
    set_attention_mask(feed_dict["attention_mask"], "padding_mask")

    quantization_config = QuantizationConfig.from_kwargs(
        quantization_method=1,  # 0 for per tensor, 1 for per channel
        quantization_output=0,  # 0 for layer, 1 for channel
        quantization_mode=1,  # maxpercentile
        percentile=0.999,  # quantization percentile
        topk_ratio=0.01,  # quantization topk
    )

    mxq_compile(
        model=model,
        save_path="stsb-bert-tiny-safetensors.mxq",
        calib_data_path="./calib",
        backend="torch",
        feed_dict=feed_dict,
        quantization_config=quantization_config,
    )

```

After running `compile_mxq.py`, you will obtain a `stsb-bert-tiny-safetensors.mxq` file.