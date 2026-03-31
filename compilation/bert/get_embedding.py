import os

import torch
from transformers import BertModel

WEIGHT_PATH = "./weights/weight_dict.pth"

model = BertModel.from_pretrained("sentence-transformers-testing/stsb-bert-tiny-safetensors", trust_remote_code=True)

word_embeddings = model.embeddings.word_embeddings.weight
token_type_embeddings = model.embeddings.token_type_embeddings.weight
position_embeddings = model.embeddings.position_embeddings.weight
layernorm_weight = model.embeddings.LayerNorm.weight
layernorm_bias = model.embeddings.LayerNorm.bias

print(word_embeddings.shape)
print(token_type_embeddings.shape)
print(position_embeddings.shape)
print(layernorm_weight.shape)
print(layernorm_bias.shape)
weight_dict = {
    "word_embeddings": word_embeddings,
    "token_type_embeddings": token_type_embeddings,
    "position_embeddings": position_embeddings,
    "layernorm_weight": layernorm_weight,
    "layernorm_bias": layernorm_bias,
}

os.makedirs(os.path.dirname(WEIGHT_PATH), exist_ok=True)
torch.save(weight_dict, WEIGHT_PATH)
