import json
import os

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

os.makedirs("./huggingface", exist_ok=True)

snapshot_download(
    repo_id="Qwen/Qwen2-VL-2B-Instruct",
    local_dir="./huggingface/",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "model-00001-of-00002.safetensors"
    ],  # optional: only download specific files
)

SOURCE_FILE = "./huggingface/model-00001-of-00002.safetensors"
TENSOR_NAME = "model.embed_tokens.weight"

# The filename "model.safetensors" is required by HuggingFace's
# PreTrainedModel.from_pretrained() convention.
# This file contains only the rotated token embedding weight,
# not the full model weights.
TARGET_FILE = "./mxq/model.safetensors"

with safe_open(SOURCE_FILE, framework="pt") as f:
    tensor_names = f.keys()
    if TENSOR_NAME not in tensor_names:
        raise RuntimeError(f"{TENSOR_NAME} not found in {SOURCE_FILE}")

    tensor = f.get_tensor(TENSOR_NAME)

# Apply the same rotation matrix used during language model MXQ compilation.
# The token embedding is not compiled into MXQ (it runs as a CPU lookup),
# so it must be pre-rotated to match the quantized model's transformed space.
head_out_ch_rotation_matrix_path = (
    "./spinWeight/Qwen2-VL-2B-Instruct_text_model/R1/global_rotation.pth"
)
head_out_ch_rotation_matrix = torch.jit.load(
    head_out_ch_rotation_matrix_path, map_location="cpu"
).state_dict()["0"]

embedding = tensor.double() @ head_out_ch_rotation_matrix

save_file({TENSOR_NAME: embedding.float()}, TARGET_FILE)
