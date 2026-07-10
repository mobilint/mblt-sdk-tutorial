import json
import os

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

if __name__ == "__main__":
    os.makedirs("./huggingface", exist_ok=True)

    REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"  # or "Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"
    TENSOR_NAME = "model.language_model.embed_tokens.weight"

    # Resolve the safetensors file that holds the embedding, regardless of model.
    index_path = "./huggingface/model.safetensors.index.json"
    snapshot_download(
        repo_id=REPO_ID,
        local_dir="./huggingface/",
        allow_patterns=["model.safetensors.index.json"],
    )

    if os.path.exists(index_path):  # sharded model (e.g. 4B/8B)
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        if TENSOR_NAME not in weight_map:
            raise RuntimeError(f"{TENSOR_NAME} not found in model.safetensors.index.json")
        target_file = weight_map[TENSOR_NAME]  # e.g. "model-00001-of-00002.safetensors"
    else:  # single-file model (e.g. 2B)
        target_file = "model.safetensors"

    snapshot_download(
        repo_id=REPO_ID,
        local_dir="./huggingface/",
        allow_patterns=[target_file],  # only the file that holds embed_tokens
    )

    SOURCE_FILE = f"./huggingface/{target_file}"

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

    # Apply the same SpinR1 rotation matrix generated during language model MXQ
    # compilation (mxq_compile_language.py writes it under this path). The token
    # embedding is not compiled into MXQ (it runs as a CPU lookup), so it must be
    # pre-rotated to match the quantized model's transformed space.
    head_out_ch_rotation_matrix_path = "./spinWeight/Qwen3-VL-4B-Instruct_text_model/R1/global_rotation.pth"
    head_out_ch_rotation_matrix = torch.jit.load(head_out_ch_rotation_matrix_path, map_location="cpu").state_dict()["0"]

    embedding = tensor.double() @ head_out_ch_rotation_matrix.double()

    save_file({TENSOR_NAME: embedding.float()}, TARGET_FILE)
