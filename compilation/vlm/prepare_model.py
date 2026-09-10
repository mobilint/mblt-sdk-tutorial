import json
import shutil
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
EMBEDDING_KEY = "model.language_model.embed_tokens.weight"
TARGET_DEVICES = ("aries-rb", "regulus-rb")


def resolve_model_ids(base_model_id: str) -> tuple[str, str]:
    """Return (RUNTIME_MODEL_ID, MODEL_NAME) derived from the base HF model id.

    Assumes the runtime template lives at ``mobilint/<name>`` for every
    ``<namespace>/<name>`` base model shipped by Qwen/Mobilint.
    """
    if "/" not in base_model_id:
        raise ValueError(f"--model-id must include a namespace, got {base_model_id!r}")
    _, name = base_model_id.split("/", 1)
    return f"mobilint/{name}", name


def load_rotation_matrix(path: Path) -> torch.Tensor:
    checkpoint = torch.jit.load(str(path), map_location="cpu")
    matrix = next(iter(checkpoint.state_dict().values()))
    return matrix.detach().to(torch.float32).contiguous()


def load_embedding(base_model_id: str) -> torch.Tensor:
    tensor_path = hf_hub_download(base_model_id, "model.safetensors")
    with safe_open(tensor_path, framework="pt") as tensors:
        key = next(name for name in tensors.keys() if name.endswith("embed_tokens.weight"))
        return tensors.get_tensor(key).to(torch.float32)


def save_rotated_embedding(base_model_id: str, rotation_path: Path, output_path: Path) -> None:
    embedding = load_embedding(base_model_id)
    rotation = load_rotation_matrix(rotation_path)
    if rotation.shape != (embedding.shape[1], embedding.shape[1]):
        raise ValueError(f"Rotation shape {tuple(rotation.shape)} does not match embedding width {embedding.shape[1]}")
    save_file({EMBEDDING_KEY: (embedding @ rotation).contiguous()}, output_path)


def patch_config(config_path: Path, target_device: str, encoder_name: str, decoder_name: str) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("mxq_path", None)

    for section_name, mxq_name in (
        ("vision_config", encoder_name),
        ("text_config", decoder_name),
    ):
        section = config[section_name]
        section["mxq_path"] = mxq_name
        section["target_device"] = target_device
        if target_device == "aries-rb":
            section["core_mode"] = "global8"
            section["target_clusters"] = [0, 1]
            section.pop("target_cores", None)
        else:
            section["core_mode"] = "single"
            section["target_cores"] = ["0:0"]
            section.pop("target_clusters", None)

    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_model(base_model_id: str, target_device: str, output_dir: Path, force: bool) -> None:
    runtime_model_id, model_name = resolve_model_ids(base_model_id)
    encoder_mxq = BASE_DIR / "mxq" / target_device / f"{model_name}_encoder.mxq"
    decoder_mxq = BASE_DIR / "mxq" / target_device / f"{model_name}_decoder.mxq"
    rotation_path = BASE_DIR / "spinWeight" / target_device / "global_rotation.pth"
    missing = [path for path in (encoder_mxq, decoder_mxq, rotation_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing compilation artifacts: {missing}")

    if output_dir.exists() and not force:
        raise FileExistsError(f"{output_dir} already exists. Use --force to replace it.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{model_name}.", dir=output_dir.parent) as temporary_dir:
        staging_dir = Path(temporary_dir)
        snapshot_download(
            repo_id=runtime_model_id,
            local_dir=staging_dir,
            ignore_patterns=["*.mxq", "*.safetensors"],
        )
        shutil.rmtree(staging_dir / ".cache")

        encoder_name = encoder_mxq.name
        decoder_name = decoder_mxq.name
        shutil.copy2(encoder_mxq, staging_dir / encoder_name)
        shutil.copy2(decoder_mxq, staging_dir / decoder_name)
        save_rotated_embedding(base_model_id, rotation_path, staging_dir / "model.safetensors")
        patch_config(staging_dir / "config.json", target_device, encoder_name, decoder_name)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(staging_dir, output_dir)

    print(f"Prepared model folder: {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Prepare a self-contained Qwen3-VL runtime model folder")
    parser.add_argument("--target-device", choices=TARGET_DEVICES, default="aries-rb")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _, model_name = resolve_model_ids(args.model_id)
    output_dir = args.output_dir or BASE_DIR / "prepared" / args.target_device / model_name
    prepare_model(args.model_id, args.target_device, output_dir, args.force)
