import argparse
import json
import os
import shutil

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file


def prepare_model_folder(
    mxq_path: str,
    embedding_path: str,
    output_folder: str,
    model_id: str,
):
    """
    Prepare a model folder for LLM MXQ inference.

    Args:
        mxq_path: Path to the compiled MXQ file
        embedding_path: Path to the embedding weight file
        output_folder: Destination folder for the prepared model
        model_id: HuggingFace model ID for config and tokenizer download
    """
    os.makedirs(output_folder, exist_ok=True)

    # Copy MXQ file
    mxq_filename = os.path.basename(mxq_path)
    print(f"Copying MXQ: {mxq_filename}")
    shutil.copy(mxq_path, os.path.join(output_folder, mxq_filename))

    # Convert embedding weight from .pt to safetensors format (expected by mblt-model-zoo)
    print(f"Converting embedding: {embedding_path} -> model.safetensors")
    embedding_tensor = torch.load(embedding_path, map_location="cpu")
    save_file({"model.embed_tokens.weight": embedding_tensor}, os.path.join(output_folder, "model.safetensors"))

    # Download config.json from HuggingFace and update MXQ path + NPU core allocation
    print(f"Downloading config from {model_id}...")
    config_path = hf_hub_download(model_id, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    config["mxq_path"] = mxq_filename
    config["_name_or_path"] = model_id

    # NPU core allocation for mblt-model-zoo inference.
    # Required when MXQ is compiled with inference_scheme="all".
    #
    # Available core modes:
    #   single: target_cores=["0:0"]    — Cluster 0, Core 0 (default)
    #           target_cores=["0:1"]    — Cluster 0, Core 1
    #           target_cores=["0:3"]    — Cluster 0, Core 3
    #           target_cores=["1:0"]    — Cluster 1, Core 0
    #   multi:   core_mode="multi",   target_clusters=[0]
    #   global4: core_mode="global4", target_clusters=[0]
    #   global8: core_mode="global8", target_clusters=[0, 1]
    config["target_cores"] = ["0:0"]

    output_config_path = os.path.join(output_folder, "config.json")
    with open(output_config_path, "w") as f:
        json.dump(config, f, indent=4)
    print("Saved config.json with NPU core allocation")

    # Download tokenizer files from HuggingFace
    print(f"Downloading tokenizer from {model_id}...")
    required_tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    optional_tokenizer_files = ["generation_config.json"]

    for filename in required_tokenizer_files:
        try:
            src = hf_hub_download(model_id, filename)
            shutil.copy(src, os.path.join(output_folder, filename))
        except Exception as e:
            raise ValueError(f"Failed to download required file '{filename}' from {model_id}: {e}")

    for filename in optional_tokenizer_files:
        try:
            src = hf_hub_download(model_id, filename)
            shutil.copy(src, os.path.join(output_folder, filename))
        except Exception:
            pass

    # Print summary
    print(f"\nModel folder prepared: {output_folder}")
    print("Contents:")
    for f in sorted(os.listdir(output_folder)):
        size = os.path.getsize(os.path.join(output_folder, f))
        if size > 1024 * 1024:
            print(f"  {f} ({size / 1024 / 1024:.2f} MB)")
        elif size > 1024:
            print(f"  {f} ({size / 1024:.2f} KB)")
        else:
            print(f"  {f} ({size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Model Folder for LLM MXQ Inference")
    parser.add_argument(
        "--mxq-path",
        type=str,
        default="../../compilation/llm/Llama-3.2-1B-Instruct.mxq",
        help="Path to the compiled MXQ file",
    )
    parser.add_argument(
        "--embedding-path",
        type=str,
        default="../../compilation/llm/embedding.pt",
        help="Path to the embedding weight file",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="./llama-mxq",
        help="Destination folder for the prepared model",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="mobilint/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID for config and tokenizer download",
    )

    args = parser.parse_args()

    if not os.path.exists(args.mxq_path):
        raise FileNotFoundError(
            f"MXQ file not found: {args.mxq_path}\n"
            "Please run the compilation tutorial first."
        )
    if not os.path.exists(args.embedding_path):
        raise FileNotFoundError(
            f"Embedding file not found: {args.embedding_path}\n"
            "Please run the compilation tutorial first."
        )

    prepare_model_folder(
        mxq_path=args.mxq_path,
        embedding_path=args.embedding_path,
        output_folder=args.output_folder,
        model_id=args.model_id,
    )

    print("\nYou can now run inference with:")
    print(
        f"  python inference_mblt_model_zoo.py --model-folder {args.output_folder} --model-id {args.model_id}"
    )
