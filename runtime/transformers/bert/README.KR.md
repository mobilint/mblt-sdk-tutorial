# Bidirectional Encoder Representations from Transformers (BERT)

이 튜토리얼은 컴파일된 BERT 모델을 사용하여 추론(Inference)을 수행하는 세부 지침을 제공합니다.

이 가이드는 `mblt-sdk-tutorial/compilation/transformers/bert/README.md`의 내용을 이어받아 진행됩니다. 모델이 성공적으로 컴파일되었으며 다음 파일들이 준비되어 있다고 가정합니다.

- `./stsb-bert-tiny-safetensors.mxq` - 컴파일된 모델 파일
- `./weight_dict.pth` - 임베딩 레이어 가중치

## 추론 실행

SBERT 모델을 실행하려면 표준 BERT 모델 인터페이스를 모방한 래퍼(Wrapper) 클래스가 필요합니다. 이 클래스는 다음과 같이 PyTorch와 `qbruntime`을 사용하여 정의됩니다.

이 클래스는 CPU에서 입력 임베딩을 처리하고, 나머지 레이어는 Mobilint NPU에서 실행합니다.

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

이 클래스를 사용하면 주어진 텍스트 쌍 사이의 유사도를 계산할 수 있습니다.

원본 모델과 결과를 비교하고 싶다면 다음 명령어를 실행하십시오.

```bash
python3 inference_original.py
```

## 성능 평가

더미 코퍼스(Dummy corpus)로 SBERT를 실행할 때 결과에 차이가 있을 수 있지만, 모델이 얼마나 잘 작동하는지 정확히 측정하기는 어렵습니다. 따라서 성능을 평가하기 위해 KorSTS 데이터셋의 모든 문장 쌍에 대한 유사도를 계산하고, 원본 점수와 추론 점수 사이의 피어슨(Pearson) 및 스피어먼(Spearman) 상관 계수를 계산합니다.

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

스크립트를 실행하려면 다음 명령어를 사용하십시오.

```bash
python3 benchmark_mxq.py
```
