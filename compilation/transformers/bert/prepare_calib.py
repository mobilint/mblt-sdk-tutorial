import argparse
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def generate_calibration(
    tokenizer_path: str,
    weight_path: str,
    output_dir: str = "./calib",
    max_calib: int = 256,
):

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    os.makedirs(output_dir, exist_ok=True)

    weight_dict = torch.load(weight_path)
    word_embeddings = torch.nn.Embedding.from_pretrained(weight_dict["word_embeddings"])
    token_type_embeddings = torch.nn.Embedding.from_pretrained(
        weight_dict["token_type_embeddings"]
    )
    position_embeddings = torch.nn.Embedding.from_pretrained(
        weight_dict["position_embeddings"]
    )
    layernorm_weight = weight_dict["layernorm_weight"]
    layernorm_bias = weight_dict["layernorm_bias"]

    try:
        dataset = load_dataset("mteb/stsbenchmark-sts", split="validation")["sentence1"]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # choose calibration from every step.

    # randomly select  max_calib data
    dataset = random.sample(dataset, max_calib)

    cur_num_calib = 0
    for i, text in enumerate(tqdm(dataset)):

        try:
            token = tokenizer(text, return_tensors="pt")
            input_ids = token["input_ids"]
            token_type_ids = token["token_type_ids"]
        except Exception as e:
            continue

        word_embed = word_embeddings(input_ids)
        token_type_embed = token_type_embeddings(token_type_ids)
        position_embed = position_embeddings(torch.arange(input_ids.shape[1]))
        embedded_text = word_embed + token_type_embed + position_embed
        embedded_text = torch.nn.functional.layer_norm(
            embedded_text,
            embedded_text.shape[-1:],
            weight=layernorm_weight,
            bias=layernorm_bias,
        )

        output_path = os.path.join(output_dir, f"{cur_num_calib}.npy")

        np.save(output_path, embedded_text.detach().cpu().numpy())
        cur_num_calib += 1


def main():
    parser = argparse.ArgumentParser(
        description="Generate calibration datasets from Wikipedia for LLM models"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="sentence-transformers-testing/stsb-bert-tiny-safetensors",
    )
    parser.add_argument("--weight-path", type=str, default="./weight_dict.pth")
    parser.add_argument("--output-dir", type=str, default="./calib")
    parser.add_argument("--max-calib", type=int, default=256)

    args = parser.parse_args()

    generate_calibration(
        tokenizer_path=args.tokenizer_path,
        weight_path=args.weight_path,
        output_dir=args.output_dir,
        max_calib=args.max_calib,
    )


if __name__ == "__main__":
    main()
