from argparse import ArgumentParser
from pathlib import Path

import torch
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import BertTokenizer
from wrapper.bert_model import BertMXQ

MXQ_FILENAME = "stsb-bert-tiny-safetensors.mxq"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path("../../../compilation/bert/bert-mxq"),
    )
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_folder)
    model = BertMXQ(args.model_folder / MXQ_FILENAME, args.model_folder)

    try:
        sts_dataset = load_dataset("mteb/stsbenchmark-sts", split="test")
        original_score = []
        inference_score = []
        pbar = tqdm(
            zip(sts_dataset["sentence1"], sts_dataset["sentence2"], sts_dataset["score"]),
            total=len(sts_dataset["sentence1"]),
        )
        for s1, s2, score in pbar:
            s1 = model(**tokenizer(s1, return_tensors="pt"))
            s2 = model(**tokenizer(s2, return_tensors="pt"))
            similarity = torch.nn.functional.cosine_similarity(s1, s2, dim=1)
            original_score.append(score)
            inference_score.append(similarity.item())

        original_score = torch.Tensor(original_score)
        inference_score = torch.Tensor(inference_score)
        pearson = pearsonr(original_score, inference_score)
        spearman = spearmanr(original_score, inference_score)
        print(f"\n=== MXQ Model Benchmark Results (STS Benchmark, {len(sts_dataset)} pairs) ===")
        print(f"Pearson correlation:  {pearson.statistic:.4f} (p={pearson.pvalue:.2e})")
        print(f"Spearman correlation: {spearman.statistic:.4f} (p={spearman.pvalue:.2e})")
        print("\nHigher correlation = better alignment with human similarity judgments")
    finally:
        model.dispose()
