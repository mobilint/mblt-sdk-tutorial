import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "proxy_llama.py",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def prepare_model(mxq_path: Path, output_folder: Path, model_id: str, revision: str) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)

    for filename in MODEL_FILES:
        source = hf_hub_download(repo_id=model_id, filename=filename, revision=revision)
        shutil.copy2(source, output_folder / filename)

    shutil.copy2(mxq_path, output_folder / mxq_path.name)

    config_path = output_folder / "config.json"
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    config["mxq_path"] = mxq_path.name
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    print(f"Prepared model folder: {output_folder}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mxq-path", type=Path, default=Path("./Llama-3.2-1B-Instruct-W8.mxq"))
    parser.add_argument("--output-folder", type=Path, default=Path("./llama-mxq-w8"))
    parser.add_argument("--model-id", default="mobilint/Llama-3.2-1B-Instruct")
    parser.add_argument("--revision", default="W8")
    args = parser.parse_args()

    if not args.mxq_path.is_file():
        raise FileNotFoundError(f"MXQ file not found: {args.mxq_path}")

    prepare_model(args.mxq_path, args.output_folder, args.model_id, args.revision)
