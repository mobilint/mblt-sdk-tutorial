from transformers import AutoModelForCausalLM
import torch

model_dir = "./Llama-3.2-1B-Instruct"  # path to the downloaded model
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
)
model.eval()
embedding_weight = model.get_input_embeddings().weight.to(
    torch.bfloat16
)  # (vocab_size, dim)

torch.save(embedding_weight, model_dir.split("/")[-1] + "_embedding_weight.pt")
