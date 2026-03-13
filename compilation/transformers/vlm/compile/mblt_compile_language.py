"""
Qwen2-VL Language Model Compilation to MBLT Format

This module handles the compilation of the Qwen2-VL language model (decoder) to MBLT
format using the qbcompiler compiler. The language model is patched with Aries 2-compatible
transformations and optimizations before compilation.

Key Transformations:
-------------------
- Pre-cached RoPE embeddings (eliminates runtime trigonometric operations)
- Last-query slicing for final layer (decode phase optimization)
- Stateful KV cache wrappers (efficient auto-regressive generation)
- Dynamic shape handling (variable sequence lengths)
"""

from typing import Dict, List, Optional, Tuple

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
    messages: List[Dict],
    output_path: str = "mblt/Qwen2-VL-2B-Instruct_text_model.mblt",
    target_device: str = "aries2",
    num_blocks: Optional[int] = None,
    ignore_weight: bool = False,
    debug: bool = True,
) -> Tuple[str, str, str]:
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

    # Prepare inputs using common utility (with image resize for OOM prevention)
    inputs = prepare_inputs(processor, messages, model.device, image_size=(224, 224))

    # ========================================================================
    # STEP 1: CAPTURE LANGUAGE MODEL INPUTS
    # ========================================================================
    print("\n[1/6] Capturing language model inputs...")

    inputs_container = DefaultInputsCaptureContainer()
    max_call_limit = 1  # Only capture the first call

    with InputCaptureCtxManager(
        model.projection, max_call_limit, inputs_container
    ) as f:
        # Run full generation to capture inputs with vision embeddings merged
        _ = model.generate(**inputs, max_new_tokens=500)

    # Extract the last captured call's inputs (after vision processing)
    feed_dict = inputs_container.captured_kwargs[-1]

    # ========================================================================
    # STEP 2: WRAP TENSORS AND MARK DYNAMIC DIMENSIONS
    # ========================================================================
    print("\n[2/6] Configuring dynamic shapes for variable sequence lengths...")

    # Wrap all tensors with metadata
    fd_inputs = {}
    for k, v in feed_dict.items():
        if isinstance(v, torch.Tensor):
            wrapped = wrap_tensor(k, v.to(model.device))
            fd_inputs[k] = wrapped
        else:
            fd_inputs[k] = v

    # Mark sequence length dimensions as dynamic
    fd_inputs["attention_mask"].src_shape[-1].set_dynamic(True)  # [B, seq_len]
    fd_inputs["position_ids"].src_shape[-1].set_dynamic(True)  # [3, B, seq_len]
    fd_inputs["inputs_embeds"].src_shape[1].set_dynamic(True)  # [B, seq_len, hidden]
    fd_inputs["cache_position"].src_shape[0].set_dynamic(True)  # [seq_len]

    # Mark attention mask as causal
    set_attention_mask(fd_inputs["attention_mask"], "causal_mask")

    print(f"   ✓ Marked attention_mask[-1] as dynamic")
    print(f"   ✓ Marked position_ids[-1] as dynamic")
    print(f"   ✓ Marked inputs_embeds[1] as dynamic")
    print(f"   ✓ Marked cache_position[0] as dynamic")
    print(f"   ✓ Set attention_mask type: causal_mask")

    # ========================================================================
    # STEP 3: CREATE PATCHED LANGUAGE MODEL
    # ========================================================================
    print("\n[3/6] Applying Aries 2 architectural patches...")

    target_model = model.projection
    target_model.language_model.rotary_emb = CachedQwen2VLTextRotaryEmbedding(
        target_model.language_model.rotary_emb
    )

    # Optionally limit number of layers for faster compilation/testing
    if num_blocks is not None:
        print(f"   ⚠ Limiting to {num_blocks} transformer blocks (for testing)")
        target_model.language_model.layers = target_model.language_model.layers[
            :num_blocks
        ]
    else:
        print(
            f"   ✓ Compiling all {len(target_model.language_model.layers)} transformer blocks"
        )
    # Pre-compute and cache RoPE embeddings for max sequence length
    target_model.language_model.rotary_emb.set_rope(feed_dict["position_ids"])

    print(f"   ✓ Applied CachedQwen2VLRotaryEmbedding (pre-cached RoPE)")
    print(f"   ✓ Applied PatchedQwen2VLSdpaAttention (last-query slicing)")
    print(f"   ✓ Applied PatchedQwen2VLDecoderLayer (final layer optimization)")
    print(f"   ✓ Pre-computed RoPE for max_seq_len=16384")

    # ========================================================================
    # STEP 4: COMPILE WITH qbcompiler PARSER
    # ========================================================================
    print(
        f"\n[4/6] Compiling to MBLT with qbcompiler parser (target: {target_device})..."
    )

    # Specify expected output format
    output_meta = {
        "type": "list",
        "keys": [0],
    }

    parser = ModelParser(
        model=target_model,
        backend="torch",  # PyTorch FX tracing
        target_device=target_device,
        yolo_decode_include=True,
    )

    # Configure parser options
    parser.cfg.allocate_to_devices = True
    parser.cfg.split_supported_concat = True

    # Run compilation
    parser.parse(
        feed_dict=fd_inputs,
        save_subgraph_type=1,
        debug=debug,
        output_meta=output_meta,
    )

    print(f"   ✓ FX graph tracing complete")
    print(f"   ✓ Applied {len(parser.model_dict.subgraphs)} graph transformations")

    md, wd = parser.get_md_wd(body_only=False)

    # ========================================================================
    # STEP 6: SERIALIZE TO MBLT FORMAT
    # ========================================================================
    print("\n[5/6] Serializing to MBLT binary format...")

    # Serialize using common utility (md and wd already obtained from parser.get_md_wd())
    serialize_to_mblt(
        md,
        wd,
        output_path,
        ignore_weight=ignore_weight,
        sort_operators=True,
    )

    # ========================================================================
    # STEP 6: VALIDATION
    # ========================================================================
    print("\n[6/6] Validating compiled model...")

    inference_values_path, comparison_path = validate_compiled_model(
        parser, model.device, output_path
    )

    # Print summary
    print_compilation_summary(
        "LANGUAGE MODEL",
        output_path,
        inference_values_path,
        comparison_path,
    )

    return output_path, inference_values_path, comparison_path


if __name__ == "__main__":
    """Example usage of language model compilation"""
    from utils import create_sample_messages, load_model_and_processor

    # Configuration
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    output_path = "mblt/Qwen2-VL-2B-Instruct_text_model.mblt"

    # Load model and processor
    model, processor = load_model_and_processor(model_name)

    # Sample input with image
    messages = create_sample_messages(
        image_url="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true",
        text_prompt="Describe this image in detail with document format.",
    )

    # Compile language model
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
