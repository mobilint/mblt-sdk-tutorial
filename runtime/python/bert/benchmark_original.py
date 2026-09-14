from argparse import ArgumentParser
from pathlib import Path

import torch
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


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

    sts_dataset = load_dataset("mteb/stsbenchmark-sts", split="test")
    original_score = []
    inference_score = []
    pbar = tqdm(
        zip(sts_dataset["sentence1"], sts_dataset["sentence2"], sts_dataset["score"]),
        total=len(sts_dataset["sentence1"]),
    )
    with torch.inference_mode():
        for s1, s2, score in pbar:
            tokens1 = tokenizer(s1, return_tensors="pt")
            tokens2 = tokenizer(s2, return_tensors="pt")
            s1 = mean_pooling(model(**tokens1).last_hidden_state, tokens1["attention_mask"])
            s2 = mean_pooling(model(**tokens2).last_hidden_state, tokens2["attention_mask"])
            similarity = torch.nn.functional.cosine_similarity(s1, s2, dim=1)
            original_score.append(score)
            inference_score.append(similarity.item())

    original_score = torch.Tensor(original_score)
    inference_score = torch.Tensor(inference_score)
    pearson = pearsonr(original_score, inference_score)
    spearman = spearmanr(original_score, inference_score)
    print(f"\n=== Original Model Benchmark Results (STS Benchmark, {len(sts_dataset)} pairs) ===")
    print(f"Pearson correlation:  {pearson.statistic:.4f} (p={pearson.pvalue:.2e})")
    print(f"Spearman correlation: {spearman.statistic:.4f} (p={spearman.pvalue:.2e})")
    print("\nHigher correlation = better alignment with human similarity judgments")
