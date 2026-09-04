import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


def generate_calibration(model_id: str, output_dir: Path, max_calib: int, seed: int) -> None:
    if max_calib <= 0:
        raise ValueError("max_calib must be greater than zero")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")

    tokenizer = BertTokenizer.from_pretrained(model_id)
    model = BertModel.from_pretrained(model_id)
    embeddings = model.embeddings.eval()
    embeddings.requires_grad_(False)
    del model

    sentences = list(load_dataset("mteb/stsbenchmark-sts", split="validation")["sentence1"])
    if max_calib > len(sentences):
        raise ValueError(f"max_calib must not exceed the validation split size ({len(sentences)})")

    output_dir.mkdir(parents=True, exist_ok=True)
    samples = random.Random(seed).sample(sentences, max_calib)

    with torch.inference_mode():
        for index, text in enumerate(tqdm(samples, desc="Generating calibration data")):
            tokens = tokenizer(text, return_tensors="pt")
            embedded_text = embeddings(
                input_ids=tokens["input_ids"],
                token_type_ids=tokens["token_type_ids"],
            )
            np.save(output_dir / f"{index}.npy", embedded_text.numpy())

    print(f"Saved {max_calib} calibration samples to {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="sentence-transformers-testing/stsb-bert-tiny-safetensors",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./calibration_data"))
    parser.add_argument("--max-calib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_calibration(args.model_id, args.output_dir, args.max_calib, args.seed)
