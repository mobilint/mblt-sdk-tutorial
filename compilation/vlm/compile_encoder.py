from argparse import ArgumentParser
from pathlib import Path

import torch
from compile_config import encoder_compile_config
from PIL import Image
from qbcompiler import mblt_compile, mxq_compile
from qbcompiler.model_dict_legacy.parser.backend.fx_hf_extensions.transformers.models.qwen3vl import (
    Qwen3VLForConditionalGenerationWrapper,
    VisionModelForQwen3VL,
    repreprocess_pixel_values,
)
from transformers import AutoProcessor

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
BASE_DIR = Path(__file__).resolve().parent
TARGET_DEVICES = ("aries-rb", "regulus-rb")


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
    parser = ArgumentParser(description="Compile the Qwen3-VL encoder")
    parser.add_argument("--target-device", choices=TARGET_DEVICES, default="aries-rb")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args()

    model_name, compiler_name = resolve_names(args.model_id)
    torch_device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen3VLForConditionalGenerationWrapper.from_pretrained(
        args.model_id,
        device_map=torch_device,
        dtype=torch.float32,
    ).eval()
    inputs = build_inputs(processor, model.device)
    images = repreprocess_pixel_values(inputs["pixel_values"], inputs["image_grid_thw"][0])
    encoder = VisionModelForQwen3VL(model.model).to(model.device).eval()
    encoder.set_grid_thw(inputs["image_grid_thw"].to(model.device))

    mblt_path = BASE_DIR / "mblt" / args.target_device / f"{compiler_name}_encoder.mblt"
    mxq_path = BASE_DIR / "mxq" / args.target_device / f"{model_name}_encoder.mxq"

    mblt_path.parent.mkdir(parents=True, exist_ok=True)
    mblt_compile(
        model=encoder,
        mblt_save_path=str(mblt_path),
        target_device=args.target_device,
        backend="torch",
        feed_dict={"images": images},
    )

    mxq_path.parent.mkdir(parents=True, exist_ok=True)
    mxq_compile(
        model=str(mblt_path),
        target_device=args.target_device,
        save_path=str(mxq_path),
        calib_data_path=str(BASE_DIR / "calibration_data/vision/npy_files.txt"),
        device="gpu" if torch_device.type == "cuda" else "cpu",
        **encoder_compile_config(args.target_device),
    )
