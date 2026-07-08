"""Shared helpers for Qwen3-VL compilation (vision encoder + language decoder).

Qwen3-VL adds a deepstack visual pathway, so ``load_model_and_processor`` wires
up the deepstack process, the projection head, and the image-feature hooks that
the compiler expects.
"""

import os

import torch
from qbcompiler.model_dict.common import WeightDict
from qbcompiler.model_dict.parser.backend.fx_hf_extensions.transformers.models.qwen3vl import (
    Projection,
    Qwen3VL_get_image_features,
    Qwen3VLForConditionalGenerationWrapper,
    Qwen3VLModel_get_image_feature,
    patched_deepstack_process,
)
from qbcompiler.model_dict.serialize import ChainedByteObj, SerializeMeta
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


def load_model_and_processor(model_name: str):
    """Load a Qwen3-VL model (float32) and processor, patched for compilation.

    The deepstack process, the projection head (language_model + lm_head), and the
    vision feature entry point are wired to the forms the compiler hooks expect.
    """
    print(f"Loading {model_name} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen3VLForConditionalGenerationWrapper.from_pretrained(
        model_name, device_map=device, torch_dtype=torch.float32
    )
    processor = AutoProcessor.from_pretrained(model_name)

    language_model = model.model.language_model
    language_model._deepstack_process = patched_deepstack_process.__get__(
        language_model, type(language_model)
    )
    model.projection = Projection(language_model, model.lm_head)

    qwen3vl_model = model.model
    qwen3vl_model.get_image_feature_class = Qwen3VLModel_get_image_feature(qwen3vl_model)
    qwen3vl_model.get_image_features = Qwen3VL_get_image_features.__get__(
        qwen3vl_model, type(qwen3vl_model)
    )
    return model, processor


def prepare_inputs(
    processor,
    messages: list[dict],
    model_device: torch.device,
    image_size: tuple[int, int] | None = None,
) -> dict:
    """Build model.generate() inputs from chat messages.

    ``image_size`` optionally resizes images for faster, lower-memory compilation.
    """
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    if image_size is not None and image_inputs:
        image_inputs = [image_inputs[0].resize(image_size)]

    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(model_device, dtype=torch.float32)

    # transformers 4.57.x emits mm_token_type_ids, which the wrapper's generate()
    # rejects; the wrapper handles image placeholders itself, so dropping it is safe.
    inputs.pop("mm_token_type_ids", None)
    return inputs


def serialize_to_mblt(
    model_dict,
    weight_dict: WeightDict,
    output_path: str,
    ignore_weight: bool = False,
) -> int:
    """Serialize a ModelDict/WeightDict pair to the MBLT binary format.
 
    Returns the size of the written file in bytes.
    """
    barr = SerializeMeta().serialize(model_dict, weight_dict, ignore_weight=ignore_weight)
 
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
 
    with open(output_path, "wb") as f:
        if isinstance(barr, bytes):
            f.write(barr)
        elif isinstance(barr, ChainedByteObj):
            barr.write(f)
 
    file_size = os.path.getsize(output_path)
    print(f"Saved {output_path} ({file_size / (1024 * 1024):.2f} MB)")
    return file_size


def create_sample_messages(image_url: str, text_prompt: str) -> list[dict]:
    """Build a single-image chat message list."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
