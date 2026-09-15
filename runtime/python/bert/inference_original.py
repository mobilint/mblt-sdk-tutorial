from argparse import ArgumentParser
from pathlib import Path

import torch
from transformers import BertModel, BertTokenizer

DUMMY_CORPUS = [
    ["A man is eating food.", "A man is eating something."],
    ["A woman is cooking food.", "A man is eating something."],
    [
        "Dubai oil prices are rising.",
        "Dubai cookies are popular.",
    ],
    ["A man is biting a dog.", "A tiger is biting a cat."],
    ["John hit Minsoo.", "Minsoo hit John."],
]


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    return (token_embeddings * expanded_mask).sum(dim=1) / expanded_mask.sum(dim=1).clamp(min=1e-9)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path("../../../compilation/bert/bert-mxq"),
    )
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_folder)
    model = BertModel.from_pretrained(args.model_folder).eval()

    print("Cosine Similarity (range: -1 to 1, higher = more similar)\n")
    with torch.inference_mode():
        for dummy_pair in DUMMY_CORPUS:
            tokens1 = tokenizer(dummy_pair[0], return_tensors="pt")
            tokens2 = tokenizer(dummy_pair[1], return_tensors="pt")
            s1 = mean_pooling(model(**tokens1).last_hidden_state, tokens1["attention_mask"])
            s2 = mean_pooling(model(**tokens2).last_hidden_state, tokens2["attention_mask"])
            similarity = torch.nn.functional.cosine_similarity(s1, s2, dim=1)
            print(f'  {similarity.item():.4f}  |  "{dummy_pair[0]}" vs "{dummy_pair[1]}"')
