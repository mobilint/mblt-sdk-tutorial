import json
import os

from huggingface_hub import snapshot_download

if __name__ == "__main__":
    os.makedirs("./huggingface", exist_ok=True)

    snapshot_download(
        repo_id="Qwen/Qwen3-VL-2B-Instruct",
        local_dir="./huggingface/",
        local_dir_use_symlinks=False,
        allow_patterns=["config.json"],  # optional: only download specific files
    )

    with open("./huggingface/config.json", encoding="utf-8") as f:
        config = json.load(f)

    config["mxq_path"] = "Qwen3-VL-2B-Instruct_text_model.mxq"
    config["vision_config"]["mxq_path"] = "Qwen3-VL-2B-Instruct_vision_transformer.mxq"
    config["architectures"] = ["MobilintQwen3VLForConditionalGeneration"]
    config["model_type"] = "mobilint-qwen3_vl"
    config["tie_word_embeddings"] = False

    with open("./mxq/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
