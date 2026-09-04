import shutil
from argparse import ArgumentParser
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_FILES = (
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
MXQ_FILENAME = "stsb-bert-tiny-safetensors.mxq"


def prepare_model(model_id: str, mxq_path: Path, output_dir: Path) -> None:
    if not mxq_path.is_file():
        raise FileNotFoundError(f"MXQ file not found: {mxq_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in MODEL_FILES:
        source = hf_hub_download(repo_id=model_id, filename=filename)
        destination = output_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    shutil.copy2(mxq_path, output_dir / MXQ_FILENAME)

    print(f"Prepared model folder: {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="sentence-transformers-testing/stsb-bert-tiny-safetensors",
    )
    parser.add_argument(
        "--mxq-path",
        type=Path,
        default=Path("./mxq/stsb-bert-tiny-safetensors.mxq"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./bert-mxq"))
    args = parser.parse_args()

    prepare_model(args.model_id, args.mxq_path, args.output_dir)
