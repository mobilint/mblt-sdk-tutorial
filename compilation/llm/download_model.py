"""Download LLM and extract embedding weights for calibration."""

from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--repo-id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--embedding-path", type=str, default="./embedding.pt")
    args = parser.parse_args()

    # Download full model to CPU (only embedding weights will be kept)
    model = AutoModelForCausalLM.from_pretrained(args.repo_id, device_map="cpu")

    # Extract embedding weight matrix: [vocab_size, embed_dim]
    embedding_layer = model.get_input_embeddings()
    weights = embedding_layer.weight.detach()
    torch.save(weights, args.embedding_path)

    print(f"Downloaded embedding weight matrix to {args.embedding_path}")
