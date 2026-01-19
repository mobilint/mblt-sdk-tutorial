# Bidirectional Encoder Representations from Transformers

이 튜토리얼은 컴파일된 BERT 모델로 인퍼런스를 실행하는 방법에 대한 자세한 지침을 제공합니다.

이 가이드는 `mblt-sdk-tutorial/compilation/transformers/bert/README.md`에서 이어집니다. 모델 컴파일을 성공적으로 마쳤으며 다음 파일들이 준비되어 있다고 가정합니다:

- `./ko-sbert-sts.mxq` - 컴파일된 모델 파일
- `./weight_dict.pth` - 임베딩 레이어 가중치

## 간단한 추론 (Simple Inference)

SBERT 모델을 실행하려면 BERT 모델 클래스의 동작을 모방하는 래퍼(wrapper)가 필요합니다. PyTorch와 `maccel`을 사용하여 다음과 같이 클래스를 정의합니다.

이 클래스는 입력 임베딩을 CPU에서 처리하고 나머지 인퍼런스 과정은 모빌린트(Mobilint)의 NPU에서 진행합니다.

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

이 클래스를 사용하여 주어진 텍스트 쌍에 대한 유사도를 계산할 수 있습니다.

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

## 성능 측정 (Measuring Performance)

SBERT를 더미 코퍼스(dummy corpus)로 실행할 때 결과값의 차이를 확인할 수는 있지만, 모델이 얼마나 잘 작동하는지 측정하기는 어렵습니다. 따라서 성능을 측정하기 위해 KorSTS 데이터셋의 모든 문장 쌍에 대해 유사도를 계산하고, 원본 점수와 인퍼런스 점수 간의 피어슨(Pearson) 및 스피어만(Spearman) 상관계수를 계산합니다.

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
