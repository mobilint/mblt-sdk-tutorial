"""Qwen2-VL Language Model Compilation to MBLT Format (via mblt_compile)."""

import argparse
import os

import torch
from qbcompiler import mblt_compile
from qbcompiler.model_dict.parser.backend.fx_hf_extensions.transformers.models.qwen2vl import (
    CachedQwen2VLTextRotaryEmbedding,
)
from qbcompiler.model_dict.parser.backend.hf.util import (
    DefaultInputsCaptureContainer,
    InputCaptureCtxManager,
)
from qbcompiler.model_dict.parser.backend.torch.object_wrapper import set_attention_mask
from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor
from utils import create_sample_messages, load_model_and_processor, prepare_inputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile Qwen2-VL language model to MBLT")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--save-path", type=str, default="mblt/Qwen2-VL-2B-Instruct_text_model.mblt")
    parser.add_argument("--target-device", type=str, required=True, help="Target NPU (e.g. aries-rb, regulus-rb)")
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model)
    messages = create_sample_messages(
        image_url="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true",
        text_prompt="Describe this image in detail with document format.",
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    inputs = prepare_inputs(processor, messages, model.device, image_size=(224, 224))

    # Capture the projection (language_model + lm_head) prefill inputs.
    container = DefaultInputsCaptureContainer()
    with InputCaptureCtxManager(model.projection, 1, container):
        model.generate(**inputs, max_new_tokens=500)
    feed_dict = container.captured_kwargs[-1]

    # Wrap inputs and mark sequence-length axes dynamic (one MBLT serves prefill + decode).
    fd_inputs = {}
    for k, v in feed_dict.items():
        fd_inputs[k] = wrap_tensor(k, v.to(model.device)) if isinstance(v, torch.Tensor) else v
    fd_inputs["attention_mask"].src_shape[-1].set_dynamic(True)
    fd_inputs["position_ids"].src_shape[-1].set_dynamic(True)
    fd_inputs["inputs_embeds"].src_shape[1].set_dynamic(True)
    fd_inputs["cache_position"].src_shape[0].set_dynamic(True)
    fd_inputs["logits_to_keep"] = 1 # keep only the last token's logits (W=1)
    set_attention_mask(fd_inputs["attention_mask"], "causal_mask")

    # Cached RoPE for the language decoder.
    target_model = model.projection
    target_model.language_model.rotary_emb = CachedQwen2VLTextRotaryEmbedding(target_model.language_model.rotary_emb)
    target_model.language_model.rotary_emb.set_rope(feed_dict["position_ids"])

    mblt_compile(
        model=target_model,
        mblt_save_path=args.save_path,
        target_device=args.target_device,
        backend="torch",
        feed_dict=fd_inputs,
        output_meta={"type": "list", "keys": [0]},
    )
    print(f"Language model compiled: {args.save_path}")
