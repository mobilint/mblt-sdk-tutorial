from argparse import ArgumentParser

import torch
from wrapper.bert_model import BertMXQ
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import BertTokenizer

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors", trust_remote_code=True
    )

    parser = ArgumentParser()
    parser.add_argument(
        "--mxq_path",
        type=str,
        default="../../compilation/bert/mxq/stsb-bert-tiny-safetensors.mxq",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default="../../compilation/bert/weights/weight_dict.pth",
    )
    args = parser.parse_args()

    model = BertMXQ(args.mxq_path, args.weight_path)

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
    pearson = pearsonr(original_score, inference_score)
    spearman = spearmanr(original_score, inference_score)
    print(f"\n=== MXQ Model Benchmark Results (STS Benchmark, {len(sts_dataset)} pairs) ===")
    print(f"Pearson correlation:  {pearson.statistic:.4f} (p={pearson.pvalue:.2e})")
    print(f"Spearman correlation: {spearman.statistic:.4f} (p={spearman.pvalue:.2e})")
    print("\nHigher correlation = closer to original model quality (1.0 = perfect match)")

    model.dispose()
