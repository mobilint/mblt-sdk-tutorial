import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch
from compile_config import TARGET_DEVICES, decoder_compile_config
from PIL import Image
from qbcompiler import mblt_compile, mxq_compile
from qbcompiler.model_dict_new.parser.backend.torch.input_capture import capture_forward_inputs
from qbcompiler.model_dict_new.parser.patcher.models.hf_models import qwen3vl
from transformers import AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_NAME = "Qwen3-VL-2B-Instruct"
COMPILER_NAME = "Qwen_Qwen3-VL-2B-Instruct"
GENERATED_ROTATION_PATH = Path("spinWeight") / f"{COMPILER_NAME}_decoder" / "R1" / "global_rotation.pth"


def load_model(device: str):
    wrapper = qwen3vl.ensure_qwen3vl_classes_loaded()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = wrapper.from_pretrained(
        MODEL_ID,
        device_map=device,
        dtype=torch.float32,
    ).eval()

    model.language_model._deepstack_process = qwen3vl.patched_deepstack_process.__get__(
        model.language_model,
        type(model.language_model),
    )
    model.projection = qwen3vl.Projection(model.language_model, model.lm_head)
    model.model.get_image_feature_class = qwen3vl.Qwen3VLModel_get_image_feature(model.model)
    model.model.get_image_features = qwen3vl.Qwen3VL_get_image_features.__get__(model.model, type(model.model))
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
    with capture_forward_inputs(model.projection, to_cpu=False) as feed_dict:
        with torch.inference_mode():
            model.generate(**build_inputs(processor, model.device), max_new_tokens=1, do_sample=False)

    position_ids = feed_dict["position_ids"]
    model.language_model.rotary_emb = qwen3vl.CachedQwen3VLTextRotaryEmbedding(model.language_model.rotary_emb)
    model.language_model.rotary_emb.set_rope(position_ids)
    feed_dict["deepstack_visual_embeds"] = qwen3vl.build_full_visual_embeds(
        feed_dict["deepstack_visual_embeds"],
        feed_dict["visual_pos_masks"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mblt_compile(
        model=model.projection,
        mblt_save_path=str(output_path),
        target_device=target_device,
        backend="torch",
        device="gpu" if torch.device(device).type == "cuda" else "cpu",
        feed_dict=dict(feed_dict),
        dynamic_axes={
            "inputs_embeds": [-2],
            "cache_position": [-1],
            "rope_deltas": [-1],
            "deepstack_visual_embeds": [-2],
        },
    )
    print(f"Saved MBLT: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Compile the Qwen3-VL decoder to MBLT and MXQ")
    parser.add_argument("--target-device", choices=TARGET_DEVICES, default="aries-rb")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    mblt_path = Path("mblt") / args.target_device / f"{COMPILER_NAME}_decoder.mblt"
    mxq_path = Path("mxq") / args.target_device / f"{MODEL_NAME}_decoder.mxq"
    calibration_path = Path("calibration_data/language/npy_files.json")
    rotation_path = Path("spinWeight") / args.target_device / "global_rotation.pth"

    build_mblt(args.target_device, args.device, mblt_path)
    mxq_path.parent.mkdir(parents=True, exist_ok=True)
    mxq_compile(
        model=str(mblt_path),
        target_device=args.target_device,
        save_path=str(mxq_path),
        calib_data_path=str(calibration_path),
        device="gpu" if torch.device(args.device).type == "cuda" else "cpu",
        **decoder_compile_config(args.target_device),
    )

    rotation_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(GENERATED_ROTATION_PATH, rotation_path)
    print(f"Saved SpinR1 matrix: {rotation_path}")
