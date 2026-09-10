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
VISION_POS_EMBED_KEY = "model.visual.pos_embed.weight"
TARGET_DEVICES = ("aries-rb", "regulus-rb")

# mblt-model-zoo hardcodes the 3-input dynamic vision contract as
# [rope, pos, folded], but qbcompiler emits inputs in graph dataflow order,
# which for VisionModelForQwen3VL is [folded, pos, rope]. Ship a proxy that
# subclasses the vision model and swaps the tuple back before infer(). This
# keeps the tutorial's prepared folder loadable with the stock model-zoo
# runtime; if a future model-zoo release adds an explicit input-order
# override the swap becomes a no-op we can remove.
DYNAMIC_PROXY_SOURCE = '''\
try:
    from mblt_model_zoo.hf_transformers.models.qwen3_vl.configuration_qwen3_vl import (
        MobilintQwen3VLConfig,
    )
    from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import (
        MobilintQwen3VLForConditionalGeneration,
        MobilintQwen3VLVisionModel,
    )
    from mblt_model_zoo.hf_transformers.models.qwen3_vl.processing_qwen3_vl import (
        MobilintQwen3VLProcessor,
    )
except ImportError:
    raise ImportError(
        "This model requires 'mblt_model_zoo' to be installed. Please run: pip install mblt_model_zoo[transformers]"
    )


_ORIG_PREPARE = MobilintQwen3VLVisionModel._prepare_dynamic_npu_inputs


def _prepare_dynamic_npu_inputs_swapped(self, hidden_states, grid):
    """Return [folded, pos, rope] to match this tutorial's compiled MXQ.

    The tutorial's compile_encoder.py --dynamic emits inputs in graph
    dataflow order, which places folded pixel values first and rope last.
    Stock mblt-model-zoo feeds [rope, pos, folded]; we swap here so the
    order matches the compiled variant.
    """
    rope, pos, folded = _ORIG_PREPARE(self, hidden_states, grid)
    return [folded, pos, rope]


MobilintQwen3VLVisionModel._prepare_dynamic_npu_inputs = _prepare_dynamic_npu_inputs_swapped

__all__ = ["MobilintQwen3VLConfig", "MobilintQwen3VLForConditionalGeneration", "MobilintQwen3VLProcessor"]
'''


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


def load_hf_tensor(base_model_id: str, key_suffix: str) -> torch.Tensor:
    """Return a float32 copy of the first tensor whose name ends with key_suffix."""
    tensor_path = hf_hub_download(base_model_id, "model.safetensors")
    with safe_open(tensor_path, framework="pt") as tensors:
        key = next(name for name in tensors.keys() if name.endswith(key_suffix))
        return tensors.get_tensor(key).to(torch.float32)


def save_runtime_weights(
    base_model_id: str,
    rotation_path: Path,
    output_path: Path,
    dynamic: bool,
) -> None:
    """Write model.safetensors: rotated embed_tokens plus visual.pos_embed if dynamic.

    mblt-model-zoo only allocates ``visual.pos_embed`` for the dynamic vision
    path, so bundling this weight into a static release would trigger a
    spurious "unused weight" warning; conversely, omitting it on a dynamic
    release yields a "MISSING: newly initialized" warning and wrong outputs.
    """
    embedding = load_hf_tensor(base_model_id, "embed_tokens.weight")
    rotation = load_rotation_matrix(rotation_path)
    if rotation.shape != (embedding.shape[1], embedding.shape[1]):
        raise ValueError(f"Rotation shape {tuple(rotation.shape)} does not match embedding width {embedding.shape[1]}")

    tensors: dict[str, torch.Tensor] = {EMBEDDING_KEY: (embedding @ rotation).contiguous()}
    if dynamic:
        tensors[VISION_POS_EMBED_KEY] = load_hf_tensor(base_model_id, "visual.pos_embed.weight").contiguous()
    save_file(tensors, output_path)


def patch_config(
    config_path: Path,
    target_device: str,
    encoder_name: str,
    decoder_name: str,
    dynamic: bool,
) -> None:
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

    # Top-level release-pairing flag that MobilintQwen3VLConfig reads. The
    # runtime reconciles it against the MXQ input counts (1/3 for vision,
    # 2/3 for text) and warns if they disagree; keeping it accurate silences
    # that warning and lets the vision submodule allocate visual.pos_embed
    # from the safetensors we shipped above.
    config["dynamic_vision"] = dynamic

    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_model(
    base_model_id: str,
    target_device: str,
    output_dir: Path,
    force: bool,
    dynamic: bool,
) -> None:
    runtime_model_id, model_name = resolve_model_ids(base_model_id)
    encoder_suffix = "_encoder_dynamic" if dynamic else "_encoder"
    decoder_suffix = "_decoder_dynamic" if dynamic else "_decoder"
    encoder_mxq = BASE_DIR / "mxq" / target_device / f"{model_name}{encoder_suffix}.mxq"
    decoder_mxq = BASE_DIR / "mxq" / target_device / f"{model_name}{decoder_suffix}.mxq"
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
        save_runtime_weights(base_model_id, rotation_path, staging_dir / "model.safetensors", dynamic)
        patch_config(staging_dir / "config.json", target_device, encoder_name, decoder_name, dynamic)
        if dynamic:
            (staging_dir / "proxy_qwen3_vl.py").write_text(DYNAMIC_PROXY_SOURCE, encoding="utf-8")

        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(staging_dir, output_dir)

    print(f"Prepared model folder: {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Prepare a self-contained Qwen3-VL runtime model folder")
    parser.add_argument("--target-device", choices=TARGET_DEVICES, default="aries-rb")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Package the dynamic MXQ pair. Reads *_encoder_dynamic.mxq / *_decoder_dynamic.mxq, "
        "bundles visual.pos_embed.weight, and sets top-level dynamic_vision=true in config.json.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _, model_name = resolve_model_ids(args.model_id)
    folder_name = f"{model_name}-dynamic" if args.dynamic else model_name
    output_dir = args.output_dir or BASE_DIR / "prepared" / args.target_device / folder_name
    prepare_model(args.model_id, args.target_device, output_dir, args.force, args.dynamic)
