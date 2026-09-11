from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image
from qbcompiler import mblt_compile, mxq_compile
from qbcompiler.model_dict.parser.patcher.models.hf_models import qwen3vl
from qbcompiler.model_dict_legacy.parser.backend.fx_hf_extensions.transformers.models import (
    qwen3vl as legacy_qwen3vl,
)
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

from compile_config import encoder_compile_config

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
BASE_DIR = Path(__file__).resolve().parent
TARGET_DEVICES = ("aries-rb", "regulus-rb")

VISION_DYNAMIC_AXES = {
    "images": [-1],
    "pos_embeds": [0],
    "cos": [-2],
    "sin": [-2],
}


class StaticVisionBlock(torch.nn.Module):
    def __init__(self, vision_block: torch.nn.Module) -> None:
        super().__init__()
        self.vision_block = vision_block

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return self.vision_block(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )


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


def fold_pixel_values(pixel_values: torch.Tensor) -> torch.Tensor:
    n, fold_in = pixel_values.shape
    return pixel_values.transpose(0, 1).reshape(1, fold_in, 1, n).contiguous()


def compute_side_inputs(visual, grid_thw: torch.Tensor):
    indices, weights = get_vision_interpolation_indices_and_weights(
        grid_thw,
        num_grid_per_side=visual.num_grid_per_side,
        mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners,
        spatial_merge_size=visual.spatial_merge_size,
    )
    pos_embeds = (visual.pos_embed(indices) * weights[:, :, None]).sum(1)
    position_ids = get_vision_position_ids(grid_thw, visual.spatial_merge_size)
    rotary = visual.rotary_pos_emb(position_ids)
    embedding = torch.cat((rotary, rotary), dim=-1)
    return pos_embeds, embedding.cos()[None, None], embedding.sin()[None, None]


def compile_static(
    args,
    model,
    inputs,
    model_name: str,
    compiler_name: str,
    torch_device: torch.device,
) -> None:
    images = legacy_qwen3vl.repreprocess_pixel_values(
        inputs["pixel_values"],
        inputs["image_grid_thw"][0],
    )
    encoder = legacy_qwen3vl.VisionModelForQwen3VL(model.model)
    encoder.model.blocks = torch.nn.ModuleList(StaticVisionBlock(block.vision_block) for block in encoder.model.blocks)
    encoder = encoder.to(torch_device).eval()
    pos_embeds, cos, sin = compute_side_inputs(
        encoder.model,
        inputs["image_grid_thw"].to(torch_device),
    )
    encoder.register_buffer("pos_embeds", pos_embeds, persistent=False)
    encoder.register_buffer("cos", cos, persistent=False)
    encoder.register_buffer("sin", sin, persistent=False)

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
        **encoder_compile_config(args.target_device, model_name, dynamic=False),
    )


def compile_dynamic(
    args,
    model,
    inputs,
    model_name: str,
    compiler_name: str,
    torch_device: torch.device,
) -> None:
    grid_thw = inputs["image_grid_thw"].to(torch_device)

    encoder = qwen3vl.VisionModelForQwen3VL(model).to(torch_device).eval()
    pos_embeds, cos, sin = compute_side_inputs(encoder.model, grid_thw)
    folded = fold_pixel_values(inputs["pixel_values"].to(torch_device).to(torch.float32))
    feed_dict = {"images": folded, "pos_embeds": pos_embeds, "cos": cos, "sin": sin}

    mblt_path = BASE_DIR / "mblt" / args.target_device / f"{compiler_name}_encoder_dynamic.mblt"
    mxq_path = BASE_DIR / "mxq" / args.target_device / f"{model_name}_encoder_dynamic.mxq"

    mblt_path.parent.mkdir(parents=True, exist_ok=True)
    mblt_compile(
        model=encoder,
        mblt_save_path=str(mblt_path),
        target_device=args.target_device,
        backend="torch",
        feed_dict=feed_dict,
        dynamic_axes=VISION_DYNAMIC_AXES,
    )

    mxq_path.parent.mkdir(parents=True, exist_ok=True)
    mxq_compile(
        model=str(mblt_path),
        target_device=args.target_device,
        save_path=str(mxq_path),
        calib_data_path=str(BASE_DIR / "calibration_data/vision/npy_files.json"),
        device="gpu" if torch_device.type == "cuda" else "cpu",
        **encoder_compile_config(args.target_device, model_name, dynamic=True),
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Compile the Qwen3-VL encoder")
    parser.add_argument("--target-device", choices=TARGET_DEVICES, default="aries-rb")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    model_name, compiler_name = resolve_names(args.model_id)
    torch_device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id,
            dtype=torch.float32,
        )
        .to(torch_device)
        .eval()
    )
    inputs = build_inputs(processor, torch_device)

    if args.dynamic:
        compile_dynamic(args, model, inputs, model_name, compiler_name, torch_device)
    else:
        compile_static(args, model, inputs, model_name, compiler_name, torch_device)
