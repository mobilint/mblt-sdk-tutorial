from argparse import ArgumentParser
from pathlib import Path

import torch
from compile_config import TARGET_DEVICES, encoder_compile_config
from PIL import Image
from qbcompiler import mblt_compile, mxq_compile
from qbcompiler.model_dict_new.parser.patcher.models.hf_models import qwen3vl
from transformers import AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_NAME = "Qwen3-VL-2B-Instruct"
COMPILER_NAME = "Qwen_Qwen3-VL-2B-Instruct"


def load_model(device: str):
    wrapper = qwen3vl.ensure_qwen3vl_classes_loaded()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = wrapper.from_pretrained(
        MODEL_ID,
        device_map=device,
        dtype=torch.float32,
    ).eval()
    return processor, model


def build_inputs(processor, device):
    with Image.open("images/image_0000.jpg") as source:
        image = source.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
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


def build_mblt(target_device: str, device: str, output_path: Path) -> None:
    processor, model = load_model(device)
    inputs = build_inputs(processor, model.device)
    images = qwen3vl.fold_pixel_values(inputs["pixel_values"])
    vision_model = qwen3vl.VisionModelForQwen3VL(model).to(model.device).eval()
    vision_model.set_grid_thw(inputs["image_grid_thw"].to(model.device))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mblt_compile(
        model=vision_model,
        mblt_save_path=str(output_path),
        target_device=target_device,
        backend="torch",
        device="gpu" if torch.device(device).type == "cuda" else "cpu",
        feed_dict={"images": images},
    )
    print(f"Saved MBLT: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Compile the Qwen3-VL encoder to MBLT and MXQ")
    parser.add_argument("--target-device", choices=TARGET_DEVICES, default="aries-rb")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    mblt_path = Path("mblt") / args.target_device / f"{COMPILER_NAME}_encoder.mblt"
    mxq_path = Path("mxq") / args.target_device / f"{MODEL_NAME}_encoder.mxq"
    calibration_path = Path("calibration_data/vision/npy_files.txt")

    build_mblt(args.target_device, args.device, mblt_path)
    mxq_path.parent.mkdir(parents=True, exist_ok=True)
    mxq_compile(
        model=str(mblt_path),
        target_device=args.target_device,
        save_path=str(mxq_path),
        calib_data_path=str(calibration_path),
        device="gpu" if torch.device(args.device).type == "cuda" else "cpu",
        **encoder_compile_config(args.target_device),
    )
