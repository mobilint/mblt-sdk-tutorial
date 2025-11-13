"""
Qwen2-VL Vision Encoder Compilation to MBLT Format

This module handles the compilation of the Qwen2-VL vision encoder to MBLT format
using the qubee compiler. The vision encoder is patched with Aries 2-compatible
transformations before compilation.

Key Transformations:
-------------------
- 3D convolution → 2D convolution (NPU optimization)
- Split QKV projection (better parallelization)
- Pre-computed RoPE embeddings (eliminates runtime computation)
- Merged patchify operation (reduces memory transfers)
"""

import torch
from typing import Dict, Tuple, List

from qubee.model_dict.common import WeightDict, DataFormat, LayerType
from qubee.model_dict.parser.backend.hf.util import (
    DefaultInputsCaptureContainer,
    InputCaptureCtxManager,
)
from qubee.model_dict.parser.backend.torch.util import wrap_tensor
from qubee.model_dict.parser.parser import ModelParser

from qubee.model_dict.parser.backend.fx_hf_extensions.transformers.models.qwen2vl import (
    repreprocess_pixel_values,
    VisionModelForQwen2VL,
)

from utils import (
    prepare_inputs,
    serialize_to_mblt,
    validate_compiled_model,
    print_compilation_summary,
)


def compile_vision_encoder(
    model,
    processor,
    messages: List[Dict],
    image_size: Tuple[int, int] = (224, 224),
    output_path: str = "qwen2vl_vision.mblt",
    target_device: str = "aries2",
    ignore_weight: bool = False,
    debug: bool = True,
) -> Tuple[str, str, str]:
    """
    Compile Qwen2-VL vision encoder to MBLT format.

    This function:
    1. Captures vision encoder inputs during a sample inference
    2. Reprocesses pixel values to Aries 2-compatible format
    3. Applies architectural patches (3D→2D conv, split QKV, etc.)
    4. Compiles the model using qubee ModelParser
    5. Serializes to MBLT binary format
    6. Validates output by comparing with original model

    Args:
        model: Qwen2VLForConditionalGeneration model
        processor: AutoProcessor for Qwen2-VL
        messages: Sample messages for input capture
        image_size: Target image size for preprocessing
        output_path: Path to save the compiled MBLT file
        target_device: Target device architecture (default: "aries2")
        ignore_weight: If True, don't save weights (graph structure only)
        debug: Enable debug logging during compilation

    Returns:
        Tuple of (mblt_path, inference_values_path, comparison_path)
    """

    print("=" * 80)
    print("VISION ENCODER COMPILATION")
    print("=" * 80)

    # Prepare inputs using common utility
    inputs = prepare_inputs(processor, messages, model.device, image_size)

    # ========================================================================
    # STEP 1: CAPTURE VISION ENCODER INPUTS
    # ========================================================================
    print("\n[1/6] Capturing vision encoder inputs...")

    inputs_container = DefaultInputsCaptureContainer()
    max_call_limit = 1  # Only capture the first call

    with InputCaptureCtxManager(model.visual, max_call_limit, inputs_container) as f:
        # Run a short generation to trigger vision encoder call
        _ = model.generate(**inputs, max_new_tokens=20)

    # Extract captured inputs
    pixel_values = inputs_container.captured_args[0][0]
    grid_thw = inputs_container.captured_kwargs[0]["grid_thw"]

    print(f"   ✓ Captured pixel_values: {pixel_values.shape}")
    print(f"   ✓ Captured grid_thw: {grid_thw}")

    # ========================================================================
    # STEP 2: REPROCESS PIXEL VALUES FOR ARIES 2
    # ========================================================================
    print("\n[2/6] Reprocessing pixel values to Aries 2 format...")

    # Convert from HF format to Aries 2-friendly format
    # Original: [num_patches, channels*patch_size^2]
    # Aries 2: [batch, channels*temporal, height, width]
    images = repreprocess_pixel_values(pixel_values, grid_thw[0])

    print(f"   ✓ Reprocessed shape: {images.shape}")

    # Wrap inputs with metadata for FX tracing
    fd_inputs = {"images": wrap_tensor("images", images.to(model.device))}
    grid_thw = grid_thw.to(model.device)

    # ========================================================================
    # STEP 3: CREATE PATCHED VISION MODEL
    # ========================================================================
    print("\n[3/6] Applying Aries 2 architectural patches...")

    # VisionModelForQwen2VL applies:
    # - 3D→2D conv transformation (NPU-optimized conv2d)
    # - Split QKV projection (better parallelization)
    # - Pre-computed RoPE embeddings (eliminates trig ops)
    # - Merged patchify operation (reduces memory transfers)
    vision_model = VisionModelForQwen2VL(model)
    vision_model.set_grid_thw(grid_thw)  # Pre-compute RoPE embeddings
    vision_model.to(model.device)

    print(f"   ✓ Applied PatchedPatchEmbed (3D→2D conv)")
    print(f"   ✓ Applied PatchedVisionSdpaAttention (split QKV)")
    print(f"   ✓ Applied PatchedPatchMerger")
    print(f"   ✓ Pre-computed RoPE embeddings for grid {grid_thw[0].tolist()}")

    # ========================================================================
    # STEP 4: COMPILE WITH QUBEE PARSER
    # ========================================================================
    print(f"\n[4/6] Compiling to MBLT with qubee parser (target: {target_device})...")

    parser = ModelParser(
        model=vision_model,
        backend="torch",  # PyTorch FX tracing
        target_device=target_device,
        yolo_decode_include=True,
    )

    # Configure parser options
    parser.cfg.allocate_to_devices = True  # Enable Aries 2/CPU partitioning
    parser.cfg.split_supported_concat = True  # Split concatenations for parallelism

    # Run compilation
    parser.parse(
        feed_dict=fd_inputs,
        save_subgraph_type=1,  # Save subgraph intermediate results
        debug=debug,
    )

    print(f"   ✓ FX graph tracing complete")
    print(f"   ✓ Applied {len(parser.model_dict.subgraphs)} graph transformations")

    # ========================================================================
    # STEP 5: SERIALIZE TO MBLT FORMAT
    # ========================================================================
    print("\n[5/6] Serializing to MBLT binary format...")

    # Extract ModelDict and WeightDict using qubee's official API
    md, wd = parser.get_md_wd(body_only=False)

    # Set data format for all input constants to NHWC (Aries 2-friendly)
    for sg in md.subgraphs:
        for op in sg.operators:
            if op.layertype == LayerType.InputConstant:
                sg.activations[op.options.outputs[0]].dataformat = DataFormat.NHWC

    # Serialize using common utility
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
        "VISION ENCODER",
        output_path,
        inference_values_path,
        comparison_path,
    )

    return output_path, inference_values_path, comparison_path


if __name__ == "__main__":
    """Example usage of vision encoder compilation"""
    from utils import load_model_and_processor, create_sample_messages

    # Configuration
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    output_path = "./qwen2vl_vision.mblt"

    # Load model and processor
    model, processor = load_model_and_processor(model_name)

    # Sample input with image
    messages = create_sample_messages(
        image_url="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg?download=true",
        text_prompt="Describe this image in detail with document format.",
    )

    # Compile vision encoder
    compile_vision_encoder(
        model=model,
        processor=processor,
        messages=messages,
        image_size=(224, 224),
        output_path=output_path,
        target_device="aries2",
        ignore_weight=False,
        debug=True,
    )
