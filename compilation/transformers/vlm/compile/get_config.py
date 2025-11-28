import json
import os

from huggingface_hub import snapshot_download

os.makedirs("./huggingface", exist_ok=True)

snapshot_download(
    repo_id="Qwen/Qwen2-VL-2B-Instruct",
    local_dir="./huggingface/",
    local_dir_use_symlinks=False,
    allow_patterns=["config.json"],  # optional: only download specific files
)

with open("./huggingface/config.json", encoding="utf-8") as f:
    config = json.load(f)

config["mxq_path"] = "Qwen2-VL-2B-Instruct_text_model.mxq"
config["vision_config"]["mxq_path"] = "Qwen2-VL-2B-Instruct_vision_transformer.mxq"
config["architectures"] = ["MobilintQwen2VLForConditionalGeneration"]
config["max_position_embeddings"] = 32768
config["model_type"] = "mobilint-qwen2_vl"
config["sliding_window"] = 32768
config["tie_word_embeddings"] = True

with open("./mxq/config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
