from argparse import ArgumentParser
from pathlib import Path

import torch
from compile_config import encoder_compile_config
from PIL import Image
from qbcompiler import mblt_compile, mxq_compile
from qbcompiler.model_dict.parser.patcher.models.hf_models.qwen3vl import (
    VisionModelForQwen3VL as DynamicVisionModelForQwen3VL,
)
from qbcompiler.model_dict_legacy.parser.backend.fx_hf_extensions.transformers.models.qwen3vl import (
    Qwen3VLForConditionalGenerationWrapper,
    VisionModelForQwen3VL,
    repreprocess_pixel_values,
)
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
BASE_DIR = Path(__file__).resolve().parent
TARGET_DEVICES = ("aries-rb", "regulus-rb")

# Vision N-axis (patch count) marked dynamic per graph input. The V2 dispatch
# is the path where dynamic_axes actually flows through to the compiled MXQ.
VISION_DYNAMIC_AXES = {
    "images": [-1],
    "pos_embeds": [0],
    "cos": [-2],
    "sin": [-2],
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


def fold_pixel_values(pixel_values: torch.Tensor) -> torch.Tensor:
    """Fold ``(N, fold_in)`` processor pixel_values into ``(1, fold_in, 1, N)``.

    The dynamic vision graph's patch_embed is a 1x1 conv over this layout, so
    the trailing N is the only size-varying axis. Must match the fold used by
    generate_calibration_data.py --dynamic and mblt-model-zoo's runtime.
    """
    n, fold_in = pixel_values.shape
    return pixel_values.transpose(0, 1).reshape(1, fold_in, 1, n).contiguous()


def compile_static(args, model_name: str, compiler_name: str, torch_device: torch.device) -> None:
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
        **encoder_compile_config(args.target_device, dynamic=False),
    )


def compile_dynamic(args, model_name: str, compiler_name: str, torch_device: torch.device) -> None:
    # V2 dispatch: pos_embeds / cos / sin are graph inputs (not InputConstants),
    # so the parsed MXQ has 4 inputs and the compiled MXQ collapses them to
    # 3 inputs after the quantizer merges cos+sin into a single rotateTensor.
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
    grid_thw = inputs["image_grid_thw"].to(torch_device)

    encoder = DynamicVisionModelForQwen3VL(model).to(torch_device).eval()
    pos_embeds, cos, sin = encoder.compute_side_inputs(grid_thw)
    folded = fold_pixel_values(inputs["pixel_values"].to(torch_device).to(torch.float32))
    # Input order matches mblt-model-zoo's _prepare_dynamic_npu_inputs:
    # [rope (cos+sin merged), pos_embeds, folded images]. Reordering the
    # feed_dict here reorders inputs in the compiled MXQ; feeding in a
    # different order at runtime trips qbruntime's variant-matching guard.
    feed_dict = {"cos": cos, "sin": sin, "pos_embeds": pos_embeds, "images": folded}

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
        **encoder_compile_config(args.target_device, dynamic=True),
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Compile the Qwen3-VL encoder")
    parser.add_argument("--target-device", choices=TARGET_DEVICES, default="aries-rb")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Compile with V2 dispatch and dynamic N axis. The compiled MXQ takes 3 inputs "
        "(folded pixel values, pos_embeds, packed rope) and pairs with a --dynamic decoder.",
    )
    args = parser.parse_args()

    model_name, compiler_name = resolve_names(args.model_id)
    torch_device = torch.device(args.device)

    if args.dynamic:
        compile_dynamic(args, model_name, compiler_name, torch_device)
    else:
        compile_static(args, model_name, compiler_name, torch_device)
