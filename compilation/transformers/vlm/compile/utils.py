"""
Utility functions for Qwen2-VL model compilation.

This module provides common functionality used across vision and language
model compilation pipelines.
"""

import torch
from typing import Dict, List, Tuple, Optional
from qwen_vl_utils import process_vision_info

from qubee.model_dict.common import WeightDict
from qubee.model_dict.serialize import SerializeMeta, ChainedByteObj


def load_model_and_processor(model_name: str):
    """
    Load Qwen2-VL model and processor from HuggingFace.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Tuple of (model, processor)
    """
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print(f"Loading model and processor from {model_name}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    print("✓ Model and processor loaded successfully")

    return model, processor


def prepare_inputs(
    processor,
    messages: List[Dict],
    model_device: torch.device,
    image_size: Optional[Tuple[int, int]] = None,
) -> Dict:
    """
    Prepare inputs for model inference from messages.

    Args:
        processor: Qwen2-VL processor
        messages: List of message dictionaries with user prompts
        model_device: Device where the model is located
        image_size: Optional tuple (height, width) to resize images

    Returns:
        Dictionary of processed inputs ready for model.generate()
    """
    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process vision inputs (images/videos)
    image_inputs, video_inputs = process_vision_info(messages)

    # Resize images if specified (for faster compilation/lower memory)
    if image_size is not None and image_inputs:
        image_inputs = [image_inputs[0].resize(image_size)]

    # Tokenize and process
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move to model device
    inputs = inputs.to(model_device)

    return inputs


def serialize_to_mblt(
    model_dict,
    weight_dict: WeightDict,
    output_path: str,
    ignore_weight: bool = False,
    sort_operators: bool = True,
) -> int:
    """
    Serialize ModelDict and WeightDict to MBLT binary format.

    Args:
        model_dict: Compiled ModelDict from parser
        weight_dict: Dictionary from parser._weight_dict
        output_path: Path to save the MBLT file
        ignore_weight: If True, don't save weights (graph structure only)
        sort_operators: Whether to sort operators topologically

    Returns:
        Size of serialized file in bytes
    """
    print("Serializing to MBLT binary format...")

    # Sort operators in topological order
    if sort_operators:
        for sg in model_dict.subgraphs:
            sg.sort_operators()

    # Serialize to binary format
    meta = SerializeMeta()
    barr = meta.serialize(model_dict, weight_dict, ignore_weight=ignore_weight)

    # Write to file
    with open(output_path, "wb") as f:
        if isinstance(barr, bytes):
            f.write(barr)
        elif isinstance(barr, ChainedByteObj):
            barr.write(f)

    # Get file size from the written file (more reliable than len(barr))
    import os
    file_size = os.path.getsize(output_path)
    file_size_mb = file_size / (1024 * 1024)

    print(f"   ✓ Saved to: {output_path}")
    print(f"   ✓ File size: {file_size_mb:.2f} MB")
    print(f"   ✓ Format: MBLT (Mobilint Binary Layout)")

    return file_size


def validate_compiled_model(
    parser,
    model_device: torch.device,
    output_path: str,
) -> Tuple[str, str]:
    """
    Validate compiled model by running inference and comparing with original.

    Args:
        parser: ModelParser instance after compilation
        model_device: Device where the model is located
        output_path: Base path for output files

    Returns:
        Tuple of (inference_values_path, comparison_path)
    """
    print("Validating compiled model...")

    # Generate output paths
    base_name = output_path.rsplit(".", 1)[0]
    inference_values_path = f"{base_name}.infer"
    comparison_path = f"{base_name}.json"

    # Run inference and compare with original model
    parser.run_inference_and_compare_output_value(
        device=model_device.type if hasattr(model_device, 'type') else str(model_device),
        save_all_inference_outputs=True,
        inference_all_outputs_write_path=inference_values_path,
        compare_result_output_path=comparison_path,
        use_value_dict_data_for_next_input=False,
        quantizer_save=True,
    )

    print(f"   ✓ Validation complete")
    print(f"   ✓ Inference values saved to: {inference_values_path}")
    print(f"   ✓ Comparison results saved to: {comparison_path}")

    return inference_values_path, comparison_path


def print_compilation_summary(
    component_name: str,
    output_path: str,
    inference_values_path: str,
    comparison_path: str,
):
    """
    Print a summary of compilation results.

    Args:
        component_name: Name of the component (e.g., "VISION ENCODER", "LANGUAGE MODEL")
        output_path: Path to the MBLT file
        inference_values_path: Path to inference values file
        comparison_path: Path to comparison results file
    """
    print("\n" + "=" * 80)
    print(f"{component_name} COMPILATION COMPLETE")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  - MBLT model: {output_path}")
    print(f"  - Inference values: {inference_values_path}")
    print(f"  - Comparison results: {comparison_path}")
    print(f"\nNext steps:")
    print(f"  - Compile MBLT → MXQ for Aries 2 deployment")
    print(f"  - Load MBLT model for runtime inference")


def create_sample_messages(image_url: str, text_prompt: str) -> List[Dict]:
    """
    Create sample messages for model input.

    Args:
        image_url: URL or path to image
        text_prompt: Text prompt for the model

    Returns:
        List of message dictionaries
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
