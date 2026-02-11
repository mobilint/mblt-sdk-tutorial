# Bidirectional Encoder Representations from Transformers (BERT)

This tutorial provides detailed instructions for performing inference with the compiled BERT model.

This guide follows `mblt-sdk-tutorial/compilation/transformers/bert/README.md`. We assume you have successfully compiled the model and have the following files ready:

- `./stsb-bert-tiny-safetensors.mxq` - Compiled model file
- `./weight_dict.pth` - Embedding layer weights

## Running Inference

To run the SBERT model, we use a wrapper class that mimics the standard BERT model interface. The class is defined using PyTorch and `qbruntime` as follows.

This class processes the input embeddings on the CPU and executes the remaining layers on the Mobilint NPU.

```python

import torch
import qbruntime


class BertMXQModel(torch.nn.Module):
    def __init__(self, mxq_path, weight_path):
        super().__init__()

        weight_dict = torch.load(weight_path)
        self.word_embeddings = torch.nn.Embedding.from_pretrained(
            weight_dict["word_embeddings"]
        )
        self.token_type_embeddings = torch.nn.Embedding.from_pretrained(
            weight_dict["token_type_embeddings"]
        )
        self.position_embeddings = torch.nn.Embedding.from_pretrained(
            weight_dict["position_embeddings"]
        )
        layernorm_weight = weight_dict["layernorm_weight"]
        layernorm_bias = weight_dict["layernorm_bias"]

        self.layernorm = torch.nn.LayerNorm(
            layernorm_weight.shape[0],
            eps=1e-12,
        )
        self.layernorm.weight.data = layernorm_weight
        self.layernorm.bias.data = layernorm_bias

        self.acc = qbruntime.Accelerator()
        mc = qbruntime.ModelConfig()
        self.model = qbruntime.Model(mxq_path, mc)
        self.model.launch(self.acc)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ):

        word_embed = self.word_embeddings(input_ids)
        token_type_embed = self.token_type_embeddings(token_type_ids)
        position_embed = self.position_embeddings(torch.arange(input_ids.shape[1]))
        embedded_text = word_embed + token_type_embed + position_embed
        embedded_text = self.layernorm(embedded_text)

        output = self.model.infer([embedded_text.cpu().numpy()])
        return torch.from_numpy(output[0]).squeeze()
```

With this class, you can compute the similarity between given text pairs.

If you want to compare the results with the original model, you can run the following command:

```bash
python3 inference_original.py
```

## Performance Evaluation

When running SBERT with a dummy corpus, you may notice variations in the results, making it difficult to assess model accuracy. Therefore, to evaluate performance, we calculate the similarity for all sentence pairs in the KorSTS dataset and compute the Pearson and Spearman correlation coefficients between the original and inference scores.

```python
from argparse import ArgumentParser

import qbruntime
import torch
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from transformers import BertTokenizer
from bertmxq import BertMXQModel

tokenizer = BertTokenizer.from_pretrained(
    "sentence-transformers-testing/stsb-bert-tiny-safetensors", trust_remote_code=True
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--mxq_path",
        type=str,
        default="../../../compilation/transformers/bert/stsb-bert-tiny-safetensors.mxq",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default="../../../compilation/transformers/bert/weight_dict.pth",
    )
    args = parser.parse_args()

    model = BertMXQModel(args.mxq_path, args.weight_path)

    sts_dataset = load_dataset("mteb/stsbenchmark-sts", split="test")
    original_score = []
    inference_score = []
    pbar = tqdm(
        zip(sts_dataset["sentence1"], sts_dataset["sentence2"], sts_dataset["score"]),
        total=len(sts_dataset["sentence1"]),
    )
    for s1, s2, score in pbar:
        with torch.no_grad():
            s1 = model(**tokenizer(s1, return_tensors="pt"))
            s2 = model(**tokenizer(s2, return_tensors="pt"))
            similarity = torch.nn.functional.cosine_similarity(s1, s2, dim=0)
        original_score.append(score)
        inference_score.append(similarity.item())

    original_score = torch.Tensor(original_score)
    inference_score = torch.Tensor(inference_score)
    ## Compute Pearson and Spearman correlation
    print("Pearson:", pearsonr(original_score, inference_score))
    print("Spearman:", spearmanr(original_score, inference_score))
```

To run the script, use the following command:

```bash
python3 benchmark_mxq.py
```
