from argparse import ArgumentParser
from pathlib import Path

import torch
from compile_config import TARGET_DEVICES, encoder_compile_config
from PIL import Image
from qbcompiler.model_dict.common import DataFormat, LayerType
from qbcompiler.model_dict.parser.backend.fx_hf_extensions.transformers.models.qwen3vl import (
    Qwen3VLForConditionalGenerationWrapper,
    VisionModelForQwen3VL,
    repreprocess_pixel_values,
)
from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor
from qbcompiler.model_dict.parser.parser import ModelParser
from qbcompiler.model_dict.serialize import ChainedByteObj, SerializeMeta
from transformers import AutoProcessor

from qbcompiler import mxq_compile

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_NAME = "Qwen3-VL-2B-Instruct"
COMPILER_NAME = "Qwen_Qwen3-VL-2B-Instruct"


def load_model(device: str):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGenerationWrapper.from_pretrained(
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
    images = repreprocess_pixel_values(inputs["pixel_values"], inputs["image_grid_thw"][0])
    vision_model = VisionModelForQwen3VL(model.model).to(model.device).eval()
    vision_model.set_grid_thw(inputs["image_grid_thw"].to(model.device))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parser = ModelParser(
        model=vision_model,
        backend="torch",
        target_device=target_device,
    )
    parser.cfg.allocate_to_devices = True
    parser.cfg.split_supported_concat = True
    parser.parse(
        feed_dict={"images": wrap_tensor("images", images.to(model.device))},
        save_subgraph_type=1,
    )
    model_dict, weight_dict = parser.get_md_wd(body_only=False)
    for subgraph in model_dict.subgraphs:
        for operator in subgraph.operators:
            if operator.layertype == LayerType.InputConstant:
                subgraph.activations[operator.options.outputs[0]].dataformat = DataFormat.NHWC

    data = SerializeMeta().serialize(model_dict, weight_dict, ignore_weight=False)
    with output_path.open("wb") as output_file:
        if isinstance(data, ChainedByteObj):
            data.write(output_file)
        else:
            output_file.write(data)
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
