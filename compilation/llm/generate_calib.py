"""Generate calibration datasets from Wikipedia for LLM quantization."""

import os
from argparse import ArgumentParser

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# LANGUAGES = ["en", "de", "fr", "es", "it", "ja", "ko", "zh"]
LANGUAGES = ["en"]


def generate_calibration(
    model_tag: str,
    embedding_path: str,
    tokenizer_path: str,
    output_dir: str = "./calib",
    min_seqlen: int = 512,
    max_seqlen: int = 2048,
    max_calib: int = 128,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    print(f"Loading embedding weights from: {embedding_path}")
    embedding_weight = torch.load(embedding_path, map_location=device)
    embedding_layer = torch.nn.Embedding.from_pretrained(embedding_weight).to(device)
    vocab_size, embed_dim = embedding_weight.shape
    print(f"Embedding shape: ({vocab_size}, {embed_dim})")

    model_name = model_tag.replace("/", "-")

    for lang in LANGUAGES:
        subset_name = f"20231101.{lang}"
        print(f"\n{'=' * 60}")
        print(f"Processing language: {lang} ({subset_name})")
        print(f"{'=' * 60}")

        # output directory: datas/{model_name}/{lang}/
        datas_dir = os.path.join(output_dir, "datas")
        model_dir = os.path.join(datas_dir, model_name)
        output_lang_dir = os.path.join(model_dir, lang)
        os.makedirs(output_lang_dir, exist_ok=True)
        print(f"Output directory: {output_lang_dir}")

        print(f"Loading Wikipedia dataset for {lang}...")
        dataset = load_dataset(
            "wikimedia/wikipedia", subset_name, split="train", streaming=True
        )

        pbar = tqdm(total=max_calib, desc=f"Calibrating ({lang})")
        cur_num_calib = 0
        for row in dataset:
            if cur_num_calib >= max_calib:
                break

            # Tokenize raw Wikipedia text into token IDs: [seq_len]
            text = row["text"]
            token_ids = tokenizer(text, return_tensors="pt")["input_ids"].squeeze().to(device)

            # Skip empty text that produces a scalar tensor after squeeze
            if token_ids.ndim == 0:
                continue

            # Convert token IDs to dense embeddings via lookup table: [seq_len, embed_dim]
            embedded_text = embedding_layer(token_ids)

            # Reshape to [1, seq_len, embed_dim] (batch dimension required for calibration)
            if embedded_text.ndim == 1:
                embedded_text = embedded_text.unsqueeze(0).unsqueeze(0)
            elif embedded_text.ndim == 2:
                embedded_text = embedded_text.unsqueeze(0)

            # Skip sequences shorter than min_seqlen, truncate those exceeding max_seqlen
            seq_len = embedded_text.shape[1]
            if seq_len < min_seqlen:
                continue
            elif seq_len > max_seqlen:
                embedded_text = embedded_text[:, :max_seqlen, :]

            output_path = os.path.join(output_lang_dir, f"inputs_embeds_{cur_num_calib}.npy")
            np.save(output_path, embedded_text.cpu().numpy())

            cur_num_calib += 1
            pbar.update(1)

        pbar.close()

    print(f"\n{'=' * 60}")
    print(f"Calibration generation completed for {model_name}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-tag",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--embedding-path",
        type=str,
        default="./embedding.pt",
    )
    parser.add_argument("--tokenizer-path", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output-dir", type=str, default="./calibration_data")
    parser.add_argument(
        "--min-seqlen",
        type=int,
        default=512,
        help="Minimum sequence length (default: 512)",
    )
    parser.add_argument(
        "--max-seqlen",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--max-calib",
        type=int,
        default=128,
        help="Number of calibration samples per language (default: 128)",
    )

    args = parser.parse_args()

    generate_calibration(
        model_tag=args.model_tag,
        embedding_path=args.embedding_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        min_seqlen=args.min_seqlen,
        max_seqlen=args.max_seqlen,
        max_calib=args.max_calib,
    )

    # Skip Python finalization to avoid PyGILState_Release crash
    # caused by lingering PyArrow/aiohttp threads from streaming datasets.
    # See: https://github.com/huggingface/datasets/issues/7357
    os._exit(0)
