"""
Qwen2-VL Language Model Compilation to MBLT Format

This module handles the compilation of the Qwen2-VL language model (decoder) to MBLT
format using the qubee compiler. The language model is patched with Aries 2-compatible
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
from qubee.model_dict.common import LayerType
from qubee.model_dict.parser.backend.fx_hf_extensions.transformers.models.qwen2vl import (
    LanguageModelForQwen2VL,
)
from qubee.model_dict.parser.backend.hf.util import (
    DefaultInputsCaptureContainer,
    InputCaptureCtxManager,
)
from qubee.model_dict.parser.backend.torch.object_wrapper import set_attention_mask
from qubee.model_dict.parser.backend.torch.util import wrap_tensor
from qubee.model_dict.parser.parser import ModelParser
from qubee.model_dict.parser.transform_operator.util import (
    check_sequential_pattern_strict_bwd,
)
from qubee.model_dict.common.layers.impl.rmsnorm import RmsNormalization

# Temporary fix for RMSNorm predict function. Will not be necessary for qubee > 0.12.0.0
def _modified_predict(self, input_: torch.Tensor, **kwargs) -> torch.Tensor:
    if not self.normalized_shape:
        variance = input_.pow(2).mean(dim=-1, keepdim=True)
        x = input_ * torch.rsqrt(variance + self.epsilon)
        return self.scale * x
    else:
        return torch.nn.functional.rms_norm(
            input_, self.normalized_shape, weight=self.scale, eps=float(self.epsilon if self.epsilon is not None else torch.finfo(x.dtype).eps)
        )
RmsNormalization._predict = _modified_predict

from utils import (
    prepare_inputs,
    print_compilation_summary,
    serialize_to_mblt,
    validate_compiled_model,
)


def configure_attention_dynamic_shapes(md, except_target_name: str):
    """
    Configure dynamic shapes for attention operators in the model graph.

    For language models with variable sequence lengths, attention operator
    activations need dynamic shape flags. This function identifies attention
    patterns and marks relevant dimensions as dynamic.

    Args:
        md: ModelDict containing the compiled graph
        except_target_name: Name of attention op that should NOT be dynamic
                           (used for final layer's last-query slicing)

    Returns:
        Number of attention operators configured
    """
    attention_ops_configured = 0

    # Iterate through all subgraphs and operators
    for sg in md.subgraphs:
        for op in sg.operators:
            # Look for MatMul operators that are part of attention
            # Attention pattern: MatMul(Softmax(MatMul(Q, K) * scale + mask), V)
            if op.layertype == LayerType.MatMul:
                left = sg.activations[op.options.inputs[0]]  # Attention scores
                right = sg.activations[op.options.inputs[1]]  # Value matrix

                # Check if left input comes from Softmax (indicates attention)
                if left.producer_op.layertype == LayerType.Softmax:
                    # Try to match attention pattern (backward from Softmax):
                    # Pattern 1: Softmax ← MultiplyConstant ← StatefulAttentionMaskWrapper ← MultiplyConstant
                    pat = check_sequential_pattern_strict_bwd(
                        left.producer_op,
                        sg,
                        LayerType.MultiplyConstant,
                        LayerType.StatefulAttentionMaskWrapper,
                        LayerType.Softmax,
                    )
                    if pat is None:
                        # Pattern 2: Softmax ← MultiplyConstant (simpler case without mask wrapper)
                        pat = check_sequential_pattern_strict_bwd(
                            left.producer_op,
                            sg,
                            LayerType.MultiplyConstant,
                            LayerType.Softmax,
                        )
                        if pat is None:
                            continue  # Not an attention pattern

                    # Determine if this attention should have dynamic sequence length
                    # The exception is for the final layer's last-query slicing
                    is_dynamic = except_target_name != op.name

                    # Handle pattern with attention mask wrapper (prefill phase)
                    if len(pat) == 3:
                        mc, smw, softmax = pat
                        mc_inact = sg.activations[mc.options.inputs[0]]
                        mc_outact = sg.activations[mc.options.outputs[0]]
                        smw_outact = sg.activations[smw.options.outputs[0]]
                        softmax_outact = sg.activations[softmax.options.outputs[0]]

                        # Verify all activations are 4D [batch, heads, seq_len, head_dim]
                        assert (
                            4
                            == len(mc_inact.get_act_src_shape())
                            == len(mc_outact.get_act_src_shape())
                            == len(smw_outact.get_act_src_shape())
                            == len(softmax_outact.get_act_src_shape())
                        )

                        # Get the preceding MatMul (Q @ K^T)
                        mm0 = mc_inact.producer_op
                        mm0_left_inact = sg.activations[mm0.options.inputs[0]]  # Q
                        mm0_right_inact = sg.activations[mm0.options.inputs[1]]  # K^T

                        # Set dynamic flags for attention score matrix (Q @ K^T output)
                        mc_inact.get_act_src_shape()[2].set_dynamic(
                            is_dynamic
                        )  # query_len
                        mc_inact.get_act_src_shape()[3].set_dynamic(True)  # key_len
                        mc_inact.shape = mc_inact.get_act_src_shape()[1:]

                        # Set dynamic flags for Query matrix
                        mm0_left_inact.get_act_src_shape()[2].set_dynamic(is_dynamic)
                        mm0_left_inact.shape = mm0_left_inact.get_act_src_shape()[1:]

                        # Set dynamic flags for Key matrix (transposed)
                        mm0_right_inact.get_act_src_shape()[3].set_dynamic(True)
                        mm0_right_inact.shape = mm0_right_inact.get_act_src_shape()[1:]

                        # Set dynamic flags for intermediate activations
                        for act in (mc_outact, smw_outact, softmax_outact):
                            act.get_act_src_shape()[2].set_dynamic(is_dynamic)
                            act.get_act_src_shape()[3].set_dynamic(True)
                            act.shape = act.get_act_src_shape()[1:]

                        # Set dynamic flags for final attention output
                        matmul_outact = sg.activations[op.options.outputs[0]]
                        matmul_outact.get_act_src_shape()[2].set_dynamic(is_dynamic)
                        matmul_outact.shape = matmul_outact.get_act_src_shape()[1:]

                        attention_ops_configured += 1

                    # Handle pattern without attention mask wrapper (decode phase)
                    elif len(pat) == 2:
                        mc, softmax = pat
                        mc_inact = sg.activations[mc.options.inputs[0]]
                        mc_outact = sg.activations[mc.options.outputs[0]]
                        softmax_outact = sg.activations[softmax.options.outputs[0]]

                        # Verify all activations are 4D
                        assert (
                            4
                            == len(mc_inact.get_act_src_shape())
                            == len(mc_outact.get_act_src_shape())
                            == len(softmax_outact.get_act_src_shape())
                        )

                        # Get the preceding MatMul (Q @ K^T)
                        mm0 = mc_inact.producer_op
                        mm0_left_inact = sg.activations[mm0.options.inputs[0]]
                        mm0_right_inact = sg.activations[mm0.options.inputs[1]]

                        # Set dynamic flags (similar to pattern 1)
                        mc_inact.get_act_src_shape()[2].set_dynamic(is_dynamic)
                        mc_inact.get_act_src_shape()[3].set_dynamic(True)
                        mc_inact.shape = mc_inact.get_act_src_shape()[1:]

                        mm0_left_inact.get_act_src_shape()[2].set_dynamic(is_dynamic)
                        mm0_left_inact.shape = mm0_left_inact.get_act_src_shape()[1:]

                        mm0_right_inact.get_act_src_shape()[3].set_dynamic(True)
                        mm0_right_inact.shape = mm0_right_inact.get_act_src_shape()[1:]

                        for act in (mc_outact, softmax_outact):
                            act.get_act_src_shape()[2].set_dynamic(is_dynamic)
                            act.get_act_src_shape()[3].set_dynamic(True)
                            act.shape = act.get_act_src_shape()[1:]

                        matmul_outact = sg.activations[op.options.outputs[0]]
                        matmul_outact.get_act_src_shape()[2].set_dynamic(is_dynamic)
                        matmul_outact.shape = matmul_outact.get_act_src_shape()[1:]

                        attention_ops_configured += 1

    return attention_ops_configured


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
    4. Compiles the model using qubee ModelParser
    5. Configures dynamic shapes for attention operators
    6. Serializes to MBLT binary format
    7. Validates output by comparing with original model

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
    print("\n[1/7] Capturing language model inputs...")

    inputs_container = DefaultInputsCaptureContainer()
    max_call_limit = 1  # Only capture the first call

    with InputCaptureCtxManager(model.model, max_call_limit, inputs_container) as f:
        # Run full generation to capture inputs with vision embeddings merged
        _ = model.generate(**inputs, max_new_tokens=500)

    # Extract the last captured call's inputs (after vision processing)
    feed_dict = inputs_container.captured_kwargs[-1]

    # ========================================================================
    # STEP 2: WRAP TENSORS AND MARK DYNAMIC DIMENSIONS
    # ========================================================================
    print("\n[2/7] Configuring dynamic shapes for variable sequence lengths...")

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
    print("\n[3/7] Applying Aries 2 architectural patches...")

    # LanguageModelForQwen2VL applies transformations
    language_model = LanguageModelForQwen2VL(model)

    # Optionally limit number of layers for faster compilation/testing
    if num_blocks is not None:
        print(f"   ⚠ Limiting to {num_blocks} transformer blocks (for testing)")
        language_model.model.layers = language_model.model.layers[:num_blocks]
    else:
        print(
            f"   ✓ Compiling all {len(language_model.model.layers)} transformer blocks"
        )

    # Pre-compute and cache RoPE embeddings for max sequence length
    language_model.model.rotary_emb.set_rope(feed_dict["position_ids"])

    print(f"   ✓ Applied CachedQwen2VLRotaryEmbedding (pre-cached RoPE)")
    print(f"   ✓ Applied PatchedQwen2VLSdpaAttention (last-query slicing)")
    print(f"   ✓ Applied PatchedQwen2VLDecoderLayer (final layer optimization)")
    print(f"   ✓ Pre-computed RoPE for max_seq_len=16384")

    # ========================================================================
    # STEP 4: COMPILE WITH QUBEE PARSER
    # ========================================================================
    print(f"\n[4/7] Compiling to MBLT with qubee parser (target: {target_device})...")

    # Specify expected output format
    output_meta = {
        "type": "dict",
        "keys": ["logits"],
    }

    parser = ModelParser(
        model=language_model,
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

    # ========================================================================
    # STEP 5: CONFIGURE DYNAMIC SHAPES FOR ATTENTION
    # ========================================================================
    print("\n[5/7] Configuring dynamic shapes for attention operators...")

    # Extract ModelDict and WeightDict using qubee's official API
    md, wd = parser.get_md_wd(body_only=False)

    # Special case: This attention operation should NOT be dynamic
    except_target_name = "fx_patched_fn0_26/slice/matmul/mul/softmax/matmul_0"

    attention_ops_configured = configure_attention_dynamic_shapes(
        md, except_target_name
    )

    print(f"   ✓ Configured {attention_ops_configured} attention operators")
    print(f"   ✓ Set dynamic shapes for Q, K, V matrices")
    print(f"   ✓ Set exception for final layer: {except_target_name}")

    # ========================================================================
    # STEP 6: SERIALIZE TO MBLT FORMAT
    # ========================================================================
    print("\n[6/7] Serializing to MBLT binary format...")

    # Serialize using common utility (md and wd already obtained from parser.get_md_wd())
    serialize_to_mblt(
        md,
        wd,
        output_path,
        ignore_weight=ignore_weight,
        sort_operators=True,
    )

    # ========================================================================
    # STEP 7: VALIDATION
    # ========================================================================
    print("\n[7/7] Validating compiled model...")

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
        debug=True,
    )
