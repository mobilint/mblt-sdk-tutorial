import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch
from compile_config import decoder_compile_config, spin_rotation_relpath
from PIL import Image
from qbcompiler import mblt_compile, mxq_compile
from qbcompiler.model_dict.parser.backend.torch.input_capture import (
    capture_forward_inputs,
)
from qbcompiler.model_dict.parser.patcher.parts import load_for_part, prepare_part
from transformers import AutoProcessor

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
BASE_DIR = Path(__file__).resolve().parent
TARGET_DEVICES = ("aries-rb", "regulus-rb")
LANGUAGE_DYNAMIC_AXES = {
    "inputs_embeds": [-2],
    "cache_position": [-1],
    "rope_deltas": [-1],
    "deepstack_visual_embeds": [-2],
}


def resolve_names(model_id: str) -> tuple[str, str]:
    """Derive (MODEL_NAME, COMPILER_NAME) from a Hugging Face model id."""
    if "/" not in model_id:
        raise ValueError(f"--model-id must include a namespace, got {model_id!r}")
    namespace, name = model_id.split("/", 1)
    return name, f"{namespace}_{name}"


def build_inputs(processor, device):
    generator = torch.Generator().manual_seed(42)
    image = Image.fromarray(torch.randint(256, (224, 224, 3), generator=generator, dtype=torch.uint8).numpy())
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device, dtype=torch.float32)


if __name__ == "__main__":
    parser = ArgumentParser(description="Compile the Qwen3-VL decoder")
    parser.add_argument("--target-device", choices=TARGET_DEVICES, default="aries-rb")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic RoPE. Promotes the cos/sin InputConstants to a runtime rope "
        "input, producing the 3-input text MXQ that pairs with a --dynamic vision encoder.",
    )
    args = parser.parse_args()

    model_name, compiler_name = resolve_names(args.model_id)
    torch_device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = load_for_part(args.model_id, "language", dtype=torch.float32, device=torch_device).eval()
    capture_target = prepare_part(model, "language").eval()
    with capture_forward_inputs(capture_target, to_cpu=False) as feed_dict:
        model.generate(**build_inputs(processor, torch_device), max_new_tokens=1, do_sample=False)
    feed_dict = dict(feed_dict)
    dynamic_axes = {name: axes for name, axes in LANGUAGE_DYNAMIC_AXES.items() if name in feed_dict}

    suffix = "_decoder_dynamic" if args.dynamic else "_decoder"
    mblt_path = BASE_DIR / "mblt" / args.target_device / f"{compiler_name}{suffix}.mblt"
    mxq_path = BASE_DIR / "mxq" / args.target_device / f"{model_name}{suffix}.mxq"
    rotation_path = BASE_DIR / spin_rotation_relpath(args.target_device, model_name, args.dynamic)
    generated_rotation_path = BASE_DIR / "spinWeight" / f"{compiler_name}{suffix}" / "R1" / "global_rotation.pth"

    mblt_path.parent.mkdir(parents=True, exist_ok=True)
    mblt_compile(
        model=model,
        model_part="language",
        mblt_save_path=str(mblt_path),
        target_device=args.target_device,
        backend="torch",
        feed_dict=feed_dict,
        dynamic_axes=dynamic_axes,
        cpu_offload=True,
    )

    mxq_path.parent.mkdir(parents=True, exist_ok=True)
    mxq_compile(
        model=str(mblt_path),
        target_device=args.target_device,
        save_path=str(mxq_path),
        calib_data_path=str(BASE_DIR / "calibration_data/language/npy_files.json"),
        device="gpu" if torch_device.type == "cuda" else "cpu",
        cpu_offload=True,
        **decoder_compile_config(args.target_device, dynamic=args.dynamic),
    )

    rotation_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated_rotation_path, rotation_path)
    print(f"Saved SpinR1 matrix: {rotation_path}")
