"""Qwen2-VL Language Model Compilation to MBLT Format"""

import torch
from qbcompiler.model_dict.parser.backend.fx_hf_extensions.transformers.models.qwen2vl import (
    CachedQwen2VLTextRotaryEmbedding,
)
from qbcompiler.model_dict.parser.backend.hf.util import (
    DefaultInputsCaptureContainer,
    InputCaptureCtxManager,
)
from qbcompiler.model_dict.parser.backend.torch.object_wrapper import set_attention_mask
from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor
from qbcompiler.model_dict.parser.parser import ModelParser
from utils import (
    prepare_inputs,
    print_compilation_summary,
    serialize_to_mblt,
    validate_compiled_model,
)


def compile_language_model(
    model,
    processor,
    messages: list[dict],
    output_path: str = "mblt/Qwen2-VL-2B-Instruct_text_model.mblt",
    target_device: str = "aries2",
    num_blocks: int | None = None,
    ignore_weight: bool = False,
    debug: bool = True,
) -> tuple[str, str, str]:
    """
    Compile Qwen2-VL language model to MBLT format.

    This function:
    1. Captures language model inputs during a sample generation
    2. Marks sequence length dimensions as dynamic
    3. Applies architectural patches (cached RoPE, KV cache, last-query slicing)
    4. Compiles the model using qbcompiler ModelParser
    5. Serializes to MBLT binary format
    6. Validates output by comparing with original model

    Args:
        model: Qwen2VLForConditionalGeneration model
        processor: AutoProcessor for Qwen2-VL
        messages: Sample messages for input capture
        output_path: Path to save the compiled MBLT file
        target_device: Target device architecture (default: "aries2")
        num_blocks: Number of transformer blocks to compile (None = all)
        ignore_weight: If True, don't save weights (graph structure only)
        debug: Enable debug logging during compilation

    Returns:
        Tuple of (mblt_path, inference_values_path, comparison_path)
    """

    print("=" * 80)
    print("LANGUAGE MODEL COMPILATION")
    print("=" * 80)

    inputs = prepare_inputs(processor, messages, model.device, image_size=(224, 224))

    # STEP 1: Capture language model inputs
    print("\n[1/6] Capturing language model inputs...")

    inputs_container = DefaultInputsCaptureContainer()
    max_call_limit = 1  # Only capture the first call

    with InputCaptureCtxManager(model.projection, max_call_limit, inputs_container):
        _ = model.generate(**inputs, max_new_tokens=500)

    feed_dict = inputs_container.captured_kwargs[-1]

    # STEP 2: Wrap tensors and mark dynamic dimensions
    print("\n[2/6] Configuring dynamic shapes for variable sequence lengths...")

    fd_inputs = {}
    for k, v in feed_dict.items():
        if isinstance(v, torch.Tensor):
            wrapped = wrap_tensor(k, v.to(model.device))
            fd_inputs[k] = wrapped
        else:
            fd_inputs[k] = v

    fd_inputs["attention_mask"].src_shape[-1].set_dynamic(True)
    fd_inputs["position_ids"].src_shape[-1].set_dynamic(True)
    fd_inputs["inputs_embeds"].src_shape[1].set_dynamic(True)
    fd_inputs["cache_position"].src_shape[0].set_dynamic(True)
    set_attention_mask(fd_inputs["attention_mask"], "causal_mask")

    # STEP 3: Create patched language model
    print("\n[3/6] Applying Aries 2 architectural patches...")

    target_model = model.projection
    target_model.language_model.rotary_emb = CachedQwen2VLTextRotaryEmbedding(
        target_model.language_model.rotary_emb
    )

    if num_blocks is not None:
        print(f"   Limiting to {num_blocks} transformer blocks (for testing)")
        target_model.language_model.layers = target_model.language_model.layers[:num_blocks]
    else:
        print(f"   Compiling all {len(target_model.language_model.layers)} transformer blocks")

    target_model.language_model.rotary_emb.set_rope(feed_dict["position_ids"])

    # STEP 4: Compile with qbcompiler parser
    print(f"\n[4/6] Compiling to MBLT with qbcompiler parser (target: {target_device})...")

    output_meta = {"type": "list", "keys": [0]}

    parser = ModelParser(
        model=target_model,
        backend="torch",
        target_device=target_device,
        # Prevents YOLO pattern detection from false-matching VLM operators
        yolo_decode_include=True,
    )

    parser.cfg.allocate_to_devices = True
    parser.cfg.split_supported_concat = True

    parser.parse(
        feed_dict=fd_inputs,
        save_subgraph_type=1,
        debug=debug,
        output_meta=output_meta,
    )

    md, wd = parser.get_md_wd(body_only=False)

    # STEP 5: Serialize to MBLT format
    print("\n[5/6] Serializing to MBLT binary format...")

    serialize_to_mblt(
        md,
        wd,
        output_path,
        ignore_weight=ignore_weight,
        sort_operators=True,
    )

    # STEP 6: Validation
    print("\n[6/6] Validating compiled model...")

    inference_values_path, comparison_path = validate_compiled_model(
        parser, model.device, output_path
    )

    print_compilation_summary(
        "LANGUAGE MODEL",
        output_path,
        inference_values_path,
        comparison_path,
    )

    return output_path, inference_values_path, comparison_path


if __name__ == "__main__":
    from utils import create_sample_messages, load_model_and_processor

    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    output_path = "mblt/Qwen2-VL-2B-Instruct_text_model.mblt"

    model, processor = load_model_and_processor(model_name)

    messages = create_sample_messages(
        image_url="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true",
        text_prompt="Describe this image in detail with document format.",
    )

    compile_language_model(
        model=model,
        processor=processor,
        messages=messages,
        output_path=output_path,
        target_device="aries2",
        num_blocks=None,
        ignore_weight=False,
        debug=False,
    )
