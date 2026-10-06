import gc
from argparse import ArgumentParser
from pathlib import Path

import torch
from compile_config import encoder_compile_config
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

# The `vision` part with `side_inputs` traces four graph inputs: the folded pixel
# values and the host-computed position embeddings, cosine and sine. All four
# share the patch-count axis N, which is marked dynamic so one MXQ accepts any
# image size the processor produces.
VISION_PART_OPTIONS = {"side_inputs": True}
VISION_DYNAMIC_AXES = {
    "images": [-1],
    "pos_embeds": [0],
    "cos": [-2],
    "sin": [-2],
}


def resolve_names(model_id: str) -> tuple[str, str]:
    if "/" not in model_id:
        raise ValueError(f"--model-id must include a namespace, got {model_id!r}")
    namespace, name = model_id.split("/", 1)
    return name, f"{namespace}_{name}"


def build_inputs(processor, device):
    generator = torch.Generator().manual_seed(42)
    pixels = torch.randint(
        256,
        (224, 224, 3),
        generator=generator,
        dtype=torch.uint8,
    ).numpy()
    image = Image.fromarray(pixels)
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
    parser = ArgumentParser(description="Compile the Qwen3-VL encoder")
    parser.add_argument("--target-device", choices=TARGET_DEVICES, default="aries-rb")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args()

    model_name, compiler_name = resolve_names(args.model_id)
    torch_device = torch.device(args.device)
    mblt_path = BASE_DIR / "mblt" / args.target_device / f"{compiler_name}_encoder.mblt"
    mxq_path = BASE_DIR / "mxq" / args.target_device / f"{model_name}_encoder.mxq"

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = load_for_part(args.model_id, "vision", dtype=torch.float32, device=torch_device).eval()
    # Capture the vision tower's real arguments (pixel values and grid) from one forward pass.
    capture_target = prepare_part(model, "vision").eval()
    with capture_forward_inputs(capture_target, to_cpu=False) as feed_dict:
        model.generate(**build_inputs(processor, torch_device), max_new_tokens=1, do_sample=False)
    feed_dict = dict(feed_dict)

    mblt_path.parent.mkdir(parents=True, exist_ok=True)
    mblt_compile(
        model=model,
        model_part="vision",
        model_part_options=VISION_PART_OPTIONS,
        mblt_save_path=str(mblt_path),
        target_device=args.target_device,
        backend="torch",
        feed_dict=feed_dict,
        dynamic_axes=VISION_DYNAMIC_AXES,
    )

    # Release tracing resources before MXQ compilation to free GPU memory.
    del feed_dict, capture_target, model, processor
    gc.collect()
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()

    mxq_path.parent.mkdir(parents=True, exist_ok=True)
    mxq_compile(
        model=str(mblt_path),
        target_device=args.target_device,
        save_path=str(mxq_path),
        calib_data_path=str(BASE_DIR / "calibration_data/vision/npy_files.json"),
        device="gpu" if torch_device.type == "cuda" else "cpu",
        **encoder_compile_config(args.target_device, model_name, str(mblt_path)),
    )
