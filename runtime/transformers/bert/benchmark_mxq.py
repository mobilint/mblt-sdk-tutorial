from argparse import ArgumentParser

import qbruntime
import torch
from bertmxq import BertMXQModel
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import BertTokenizer

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
