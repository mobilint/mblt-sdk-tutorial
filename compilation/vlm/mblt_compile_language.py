"""Compile the Qwen3-VL language decoder (HF -> MBLT) with ModelParser.

The Projection(language_model, lm_head) prefill inputs are captured during a
single generate() step, then run through ModelParser. Sequence-length axes are
marked dynamic so one MBLT serves both prefill and decode.
"""

import argparse
import os

import torch
from qbcompiler.model_dict.parser.backend.fx_hf_extensions.transformers.models.qwen3vl import (
    CachedQwen3VLTextRotaryEmbedding,
    build_full_visual_embeds,
)
from qbcompiler.model_dict.parser.backend.torch.object_wrapper import set_attention_mask
from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor
from qbcompiler.model_dict.parser.parser import ModelParser
from utils import create_sample_messages, load_model_and_processor, prepare_inputs, serialize_to_mblt

TARGET_DEVICES = ["regulus-ra", "aries-rb", "regulus-rb"]
IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true"

# Which axis of each captured input carries the dynamic sequence length.
SEQ_LAST_DIM = ("input_ids", "position_ids", "attention_mask", "cache_position", "visual_pos_masks")
SEQ_SECOND_LAST_DIM = ("inputs_embeds", "deepstack_visual_embeds")


class _CaptureDone(Exception):
    """Raised to stop generate() as soon as the projection inputs are captured."""


def install_visual_2tuple_adapter(model):
    """Expose visual.forward as a (image_embeds, deepstack) 2-tuple.

    transformers 4.57.x returns a 3-field output, but the image-feature hook
    expects the 2-tuple form used by the compiler.
    """
    visual = model.model.visual
    original_forward = visual.forward

    def adapter(hidden_states, grid_thw, **kwargs):
        out = original_forward(hidden_states, grid_thw=grid_thw, **kwargs)
        if isinstance(out, tuple):
            return (out[0], out[1]) if len(out) == 2 else (out[1], out[2])
        return out.pooler_output, out.deepstack_features

    visual.forward = adapter


def capture_projection_inputs(model, inputs):
    """Capture the first Projection.forward kwargs during a single generate step."""
    projection = model.projection
    original_forward = projection.forward
    captured = {}

    def snapshot(*args, **kwargs):
        captured.update({k: (v.clone().detach() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()})
        raise _CaptureDone

    projection.forward = snapshot
    try:
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1, do_sample=False)
    except _CaptureDone:
        pass
    finally:
        projection.forward = original_forward

    if not captured:
        raise RuntimeError("Projection hook captured no inputs.")
    return captured


def build_feed_dict(captured, device):
    """Wrap captured inputs for the parser; return (feed_dict, position_ids)."""
    # mrope position_ids has dim0 == 4 in transformers 4.57.x; set_rope expects 3 (t/h/w).
    position_ids = captured["position_ids"].to(device)
    if position_ids.shape[0] > 3:
        position_ids = position_ids[:3]

    captured["deepstack_visual_embeds"] = build_full_visual_embeds(
        captured["deepstack_visual_embeds"], captured["visual_pos_masks"]
    )

    feed_dict = {}
    for name, value in captured.items():
        if not isinstance(value, torch.Tensor):
            feed_dict[name] = value
            continue
        wrapped = wrap_tensor(name, value.to(device))
        if name in SEQ_LAST_DIM:
            wrapped.src_shape[-1].set_dynamic(True)
        elif name in SEQ_SECOND_LAST_DIM:
            wrapped.src_shape[-2].set_dynamic(True)
        feed_dict[name] = wrapped

    feed_dict["logits_to_keep"] = 1  # emit only the last token's logits (W=1)
    set_attention_mask(feed_dict["attention_mask"], "causal_mask")
    return feed_dict, position_ids


def build_parser(model, target_device):
    """Configure a torch-backend ModelParser for the language decoder."""
    parser = ModelParser(
        model=model.projection,
        backend="torch",
        target_device=target_device,
        yolo_decode_include=True,
    )
    parser.cfg.allocate_to_devices = True
    parser.cfg.split_supported_concat = True
    return parser


def main():
    args = _parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    model, processor = load_model_and_processor(args.model)
    install_visual_2tuple_adapter(model)

    messages = create_sample_messages(IMAGE_URL, "Describe this image in detail.")
    inputs = prepare_inputs(processor, messages, model.device, image_size=(224, 224))

    captured = capture_projection_inputs(model, inputs)
    feed_dict, position_ids = build_feed_dict(captured, model.device)

    language_model = model.projection.language_model
    language_model.rotary_emb = CachedQwen3VLTextRotaryEmbedding(language_model.rotary_emb)
    language_model.rotary_emb.set_rope(position_ids)

    parser = build_parser(model, args.target_device)
    parser.parse(
        feed_dict=feed_dict,
        debug=False,
        output_meta={"type": "list", "keys": [0]},
    )

    model_dict, weight_dict = parser.get_md_wd(body_only=True)
    serialize_to_mblt(model_dict, weight_dict, args.save_path)


def _parse_args():
    parser = argparse.ArgumentParser(description="Compile Qwen3-VL language decoder to MBLT")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--save-path", default="mblt/Qwen3-VL-2B-Instruct_text_model.mblt")
    parser.add_argument("--target-device", required=True, choices=TARGET_DEVICES)
    return parser.parse_args()


if __name__ == "__main__":
    main()
