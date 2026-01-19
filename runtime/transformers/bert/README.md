# Bidirectional Encoder Representations from Transformers

This tutorial provides detailed instructions for running inference with the compiled BERT model.

This guide continues from `mblt-sdk-tutorial/compilation/transformers/bert/README.md`. We assume you have successfully compiled the model and have the following files ready:

- `./ko-sbert-sts.mxq` - Compiled model file
- `./weight_dict.pth` - Embedding layer weights

## Simple Inference

To run the SBERT model, we need a wrapper that mimics the behavior of the BERT model class. We define the class using PyTorch and `maccel` as follows.

This class processes the input embedding on the CPU and proceeds with the rest of the inference on Mobilint's NPU.

```python
import torch
import maccel 

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

        self.acc = maccel.Accelerator()
        mc = maccel.ModelConfig()
        self.model = maccel.Model(mxq_path, mc)
        self.model.launch(self.acc)
        self.model.reset_cache_memory()

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

With this class, you can compute the similarity for given text pairs.

```python
from transformers import BertTokenizer

dummy_corpus = [
    ["한 남자가 음식을 먹고 있다.", "한 남자가 무언가를 먹고 있다."],
    ["한 여성이 고기를 요리하고 있다.", "한 남자가 말하고 있다."],
    [
        "두바이산 원유의 배럴당 가격이 폭증하고 있다.",
        "두바이 쫀득 쿠키가 유행하고 있다.",
    ],
    ["어떤 남자가 개한테 물려서 다쳤다.", "어떤 호랑이가 고양이에게 물려서 다쳤다."],
    ["영수가 민수를 때렸다.", "민수가 영수한테 맞았다."],
]
tokenizer = BertTokenizer.from_pretrained("jhgan/ko-sbert-sts", trust_remote_code=True)
model = BertMXQModel(".//ko-sbert-sts.mxq", "./weight_dict.pth")

if __name__ == "__main__":
    # dummy corpus STS

    for dummy_pair in dummy_corpus:
        with torch.no_grad():
            s1 = model(**tokenizer(dummy_pair[0], return_tensors="pt"))
            s2 = model(**tokenizer(dummy_pair[1], return_tensors="pt"))
            similarity = torch.nn.functional.cosine_similarity(s1, s2, dim=0)
```

## Measuring Performance

When running SBERT with the dummy corpus, you may notice differences in the results, but it is difficult to measure how well the model is working. Hence, to measure the performance, we compute the similarity for all sentence pairs in the KorSTS dataset, and compute the Pearson and Spearman correlations between the original and inference scores.

```python
if __name__ == "__main__":
    sts_dataset = load_dataset("mteb/KorSTS", split="test")
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
