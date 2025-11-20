import argparse

import torch
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--repo_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument("--embedding", type=str, default="./embedding.pt")
args = parser.parse_args()

out_file = args.embedding
repo_id = args.repo_id

model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    device_map="cpu",
)
embedding_layer = model.get_input_embeddings()
weights = embedding_layer.weight.detach().cpu()
torch.save(weights, out_file)
print(f"Downloaded embedding weight matrix to {out_file}")
