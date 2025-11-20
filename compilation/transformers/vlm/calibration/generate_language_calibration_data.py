"""
Language Model Calibration Data Generation for Quantization

This module generates calibration data for quantizing the Qwen2-VL language model (decoder).
Each sample contains only the inputs_embeds tensor needed for calibration.

Images are loaded from the images/ folder (all 100 images), and diverse prompts
are automatically generated and cycled through to ensure calibration diversity.

Output Structure:
-----------------
calibration_data/language/
├── sample_000/
│   └── inputs_embeds.npy    # Token embeddings [B, seq_len, hidden_size]
├── sample_001/
│   └── inputs_embeds.npy
├── sample_002/
│   └── inputs_embeds.npy
├── ... (100 samples total)
├── metadata.json            # Sample information and configuration
└── npy_files.txt            # List of all .npy file paths (one per line)
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from qubee.model_dict.parser.backend.hf.util import (
    DefaultInputsCaptureContainer,
    InputCaptureCtxManager,
)


def load_model_and_processor(model_name: str):
    """Load Qwen2-VL model and processor from HuggingFace."""
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

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
    image_size: Tuple[int, int] = (224, 224),
) -> Dict:
    """Prepare inputs for model inference from messages."""
    from qwen_vl_utils import process_vision_info

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process vision inputs
    image_inputs, video_inputs = process_vision_info(messages)

    # Resize images if specified
    if image_size is not None and image_inputs:
        image_inputs = [img.resize(image_size) for img in image_inputs]

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


def create_diverse_samples() -> List[Dict]:
    """
    Create diverse calibration samples covering various scenarios.
    Uses ALL local images from images/ folder with diverse prompts.

    Prompts are cycled through different types to ensure calibration diversity:
    - Short answers
    - Detailed descriptions
    - Visual reasoning
    - Counting and enumeration
    - Spatial understanding
    - Technical analysis
    - Comparison and contrast
    - Open-ended questions

    Returns:
        List of sample configurations with local image paths and prompts
    """
    import glob

    # Base path for local images
    base_path = "./images"

    # Get all PNG images in the folder
    image_files = sorted(glob.glob(f"{base_path}/*.jpg"))

    if not image_files:
        raise FileNotFoundError(f"No PNG images found in {base_path}")

    # Define diverse prompt templates that work for general images
    # These are cycled through to ensure calibration diversity
    prompt_templates = [
        ("short_answer", "What is the main subject of this image?"),
        (
            "detailed_description",
            "Describe this image in detail, including all objects, colors, textures, and spatial relationships.",
        ),
        ("object_identification", "What objects can you identify in this image?"),
        (
            "scene_understanding",
            "Describe the scene, setting, and context shown in this image.",
        ),
        (
            "visual_reasoning",
            "Analyze what is happening in this image and explain your reasoning.",
        ),
        (
            "counting",
            "Count and list all distinct objects or elements you can identify in this image.",
        ),
        (
            "spatial_reasoning",
            "Describe the spatial arrangement and positioning of elements in this image.",
        ),
        (
            "technical_description",
            "Provide a technical description of what is shown, including materials, structure, and design.",
        ),
        (
            "color_texture",
            "Describe the colors, textures, and visual patterns present in this image.",
        ),
        (
            "comparison",
            "Compare and contrast the different elements visible in this image.",
        ),
        (
            "purpose_function",
            "What is the purpose or function of the main subject in this image?",
        ),
        (
            "environment_context",
            "Describe the environment and context surrounding the main subject.",
        ),
        (
            "detailed_analysis",
            "Provide a comprehensive analysis of this image, covering all observable details and their relationships.",
        ),
        (
            "characteristics",
            "What are the key characteristics and distinctive features of what is shown?",
        ),
        ("composition", "Analyze the composition and visual structure of this image."),
        (
            "action_activity",
            "What action or activity, if any, is taking place in this image?",
        ),
        (
            "categorization",
            "What category or type does the main subject of this image belong to?",
        ),
        ("materials", "What materials or substances can you identify in this image?"),
        (
            "lighting_atmosphere",
            "Describe the lighting, shadows, and overall atmosphere of this image.",
        ),
        ("perspective", "From what perspective or viewpoint is this image captured?"),
    ]

    # Create samples for all images
    samples = []
    for idx, image_path in enumerate(image_files):
        # Cycle through prompt templates
        template_idx = idx % len(prompt_templates)
        prompt_type, prompt_text = prompt_templates[template_idx]

        # Extract filename for naming
        filename = os.path.basename(image_path)
        sample_name = f"{prompt_type}_{idx:03d}"

        samples.append(
            {
                "name": sample_name,
                "image_url": image_path,
                "prompt": prompt_text,
            }
        )

    print(f"Created {len(samples)} calibration samples from {len(image_files)} images")
    return samples


def capture_language_model_inputs(
    model,
    processor,
    sample_config: Dict,
    image_size: Tuple[int, int] = (224, 224),
    max_new_tokens: int = 500,
) -> np.ndarray:
    """
    Capture language model inputs for a single sample.

    This captures only inputs_embeds needed for calibration:
    - inputs_embeds: Token embeddings after embedding layer [1, seq_len, 1536]

    The inputs_embeds has already passed through:
    1. Text tokenization
    2. Vision encoder (for image patches)
    3. Embedding layer (token embeddings + vision features merged)

    Args:
        model: Qwen2VLForConditionalGeneration model
        processor: AutoProcessor for Qwen2-VL
        sample_config: Dictionary with 'image_url' (local file path or URL) and 'prompt'
        image_size: Target image size for preprocessing
        max_new_tokens: Maximum tokens to generate (captures longer sequences)

    Returns:
        inputs_embeds as numpy array
    """
    # Create messages from sample config
    # Note: 'image_url' can be a local file path or URL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample_config["image_url"]},
                {"type": "text", "text": sample_config["prompt"]},
            ],
        }
    ]

    # Prepare inputs
    inputs = prepare_inputs(processor, messages, model.device, image_size)

    # Capture language model inputs
    inputs_container = DefaultInputsCaptureContainer()
    max_call_limit = 1  # Capture the first call (prefill phase)

    with InputCaptureCtxManager(model.model, max_call_limit, inputs_container) as f:
        # Run generation to capture inputs with vision embeddings merged
        _ = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Extract the last captured call's inputs (after vision processing)
    captured_kwargs = inputs_container.captured_kwargs[-1]

    # Only extract inputs_embeds (the only tensor needed for calibration)
    result = {}

    # Handle inputs_embeds - it's often None and needs to be computed
    if "inputs_embeds" in captured_kwargs and isinstance(
        captured_kwargs["inputs_embeds"], torch.Tensor
    ):
        # Already have embeddings
        tensor = captured_kwargs["inputs_embeds"]
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        result["inputs_embeds"] = tensor.cpu().numpy()
        print(f"   ✓ Using pre-computed inputs_embeds")
    else:
        # Need to compute embeddings from input_ids
        print(f"   → Computing inputs_embeds from input_ids + vision features")

        # Get input_ids and vision features
        input_ids = captured_kwargs.get("input_ids")
        pixel_values = captured_kwargs.get("pixel_values")
        image_grid_thw = captured_kwargs.get("image_grid_thw")

        if input_ids is None:
            raise ValueError("Cannot compute inputs_embeds: input_ids not found")

        # Compute embeddings using the model's embed_tokens
        with torch.no_grad():
            # Find the embed_tokens layer - try different paths
            if hasattr(model.model, "embed_tokens"):
                embed_tokens = model.model.embed_tokens
            elif hasattr(model.model, "model") and hasattr(
                model.model.model, "embed_tokens"
            ):
                embed_tokens = model.model.model.embed_tokens
            elif hasattr(model, "get_input_embeddings"):
                embed_tokens = model.get_input_embeddings()
            else:
                # Try to find it by inspecting the model structure
                print(f"   DEBUG: model.model attributes: {dir(model.model)}")
                raise AttributeError("Cannot find embed_tokens layer")

            # Get text embeddings (move input_ids to model device)
            inputs_embeds = embed_tokens(input_ids.to(model.device))

            # If we have vision features, merge them
            if pixel_values is not None and image_grid_thw is not None:
                # Process vision features
                image_embeds = model.visual(
                    pixel_values.to(model.device),
                    grid_thw=image_grid_thw.to(model.device),
                )

                # Merge vision embeddings into text embeddings
                # Use the same logic as in custom_modeling_qwen2_vl.py:1689-1696
                n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                # Create mask for image tokens
                image_mask = (
                    (input_ids == model.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )

                # Convert image embeds to same dtype and device as inputs_embeds
                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )

                # Scatter image embeddings into the text embeddings at image token positions
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

                print(
                    f"   ✓ Merged {n_image_features} image features into inputs_embeds"
                )

            # Convert to float32 and move to CPU
            if inputs_embeds.dtype == torch.bfloat16:
                inputs_embeds = inputs_embeds.float()
            result["inputs_embeds"] = inputs_embeds.cpu().numpy()

        print(f"   ✓ Computed inputs_embeds from embedding layer")

    # Validate inputs_embeds shape and range
    if "inputs_embeds" in result:
        inputs_embeds = result["inputs_embeds"]
        embeds_shape = inputs_embeds.shape
        embeds_min = inputs_embeds.min()
        embeds_max = inputs_embeds.max()
        embeds_mean = inputs_embeds.mean()
        embeds_std = inputs_embeds.std()

        print(f"   ✓ inputs_embeds shape: {embeds_shape}")
        print(
            f"   ✓ inputs_embeds range: [{embeds_min:.2f}, {embeds_max:.2f}], mean: {embeds_mean:.2f}, std: {embeds_std:.2f}"
        )

        # Expected: [batch=1, seq_len=dynamic, hidden_size=1536]
        assert (
            len(embeds_shape) == 3
        ), f"inputs_embeds should be 3D, got shape {embeds_shape}"
        assert embeds_shape[0] == 1, f"batch size should be 1, got {embeds_shape[0]}"
        expected_hidden_size = 1536  # For Qwen2-VL-2B-Instruct
        if embeds_shape[2] != expected_hidden_size:
            print(
                f"   ⚠ Warning: hidden_size is {embeds_shape[2]}, expected {expected_hidden_size}"
            )
            print(f"   This may be a different model variant")

        # Warn if range seems too small (likely missing vision features)
        if abs(embeds_max) < 1.0 and abs(embeds_min) < 1.0:
            print(
                f"   ⚠ WARNING: inputs_embeds range is very small ({embeds_min:.2f} to {embeds_max:.2f})"
            )
            print(f"   ⚠ This might indicate vision features were not properly merged!")
            print(f"   ⚠ Expected range is typically much larger (e.g., -20 to 80)")

        return inputs_embeds
    else:
        raise ValueError("Failed to obtain inputs_embeds")


def generate_language_calibration_data(
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    output_dir: str = "./calibration_data/language",
    image_size: Tuple[int, int] = (224, 224),
    num_samples: int = None,
    max_new_tokens: int = 500,
) -> str:
    """
    Generate calibration data for language model quantization.

    Args:
        model_name: HuggingFace model identifier
        output_dir: Directory to save calibration data
        image_size: Target image size for preprocessing
        num_samples: Number of calibration samples (None = all available)
        max_new_tokens: Maximum tokens to generate per sample

    Returns:
        Path to calibration data directory
    """
    print("=" * 80)
    print("LANGUAGE MODEL CALIBRATION DATA GENERATION")
    print("=" * 80)

    # Load model and processor
    model, processor = load_model_and_processor(model_name)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get diverse samples
    sample_configs = create_diverse_samples()
    if num_samples is not None:
        sample_configs = sample_configs[:num_samples]

    print(f"\nGenerating calibration data for {len(sample_configs)} samples...")

    # Collect metadata
    metadata = {
        "model_name": model_name,
        "image_size": list(image_size),
        "max_new_tokens": max_new_tokens,
        "num_samples": len(sample_configs),
        "samples": [],
    }

    # List to store all npy file paths
    npy_file_paths = []

    # Process each sample
    for i, sample_config in enumerate(sample_configs):
        print(f"\n[{i+1}/{len(sample_configs)}] Processing: {sample_config['name']}")
        print(f"   Image: {sample_config['image_url']}")
        print(f"   Prompt: {sample_config['prompt']}")

        try:
            # Capture inputs (returns only inputs_embeds as numpy array)
            inputs_embeds = capture_language_model_inputs(
                model, processor, sample_config, image_size, max_new_tokens
            )

            # Create sample directory
            sample_dir = os.path.join(output_dir, f"sample_{i:03d}")
            os.makedirs(sample_dir, exist_ok=True)

            # Save only inputs_embeds.npy
            npy_path = os.path.join(sample_dir, "inputs_embeds.npy")
            np.save(npy_path, inputs_embeds)
            shape_str = (
                str(inputs_embeds.shape) + " # [batch, seq_len, hidden_size=1536]"
            )
            print(f"   ✓ Saved inputs_embeds.npy: {shape_str}")

            # Add absolute path to list
            npy_file_paths.append(os.path.abspath(npy_path))

            # Add to metadata
            metadata["samples"].append(
                {
                    "index": i,
                    "name": sample_config["name"],
                    "prompt": sample_config["prompt"],
                    "image_url": sample_config["image_url"],
                    "directory": f"sample_{i:03d}",
                    "shape": list(inputs_embeds.shape),
                }
            )

        except Exception as e:
            print(f"   ✗ Error processing sample: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Metadata saved to: {metadata_path}")

    # Save txt file listing all npy file paths
    npy_list_path = os.path.join(output_dir, "npy_files.txt")
    with open(npy_list_path, "w") as f:
        for npy_path in npy_file_paths:
            f.write(npy_path + "\n")

    print(f"✓ NPY file list saved to: {npy_list_path}")

    # Calculate total size
    total_size = sum(
        os.path.getsize(os.path.join(root, file))
        for root, dirs, files in os.walk(output_dir)
        for file in files
    )
    total_size_mb = total_size / (1024 * 1024)

    # Print summary
    print("\n" + "=" * 80)
    print("LANGUAGE CALIBRATION DATA GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - Samples collected: {len(metadata['samples'])}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Total size: {total_size_mb:.2f} MB")
    print(f"  - Image size: {image_size}")
    print(f"  - Max tokens per sample: {max_new_tokens}")
    print(f"\nStructure:")
    print(f"  {output_dir}/")
    for sample in metadata["samples"][:3]:  # Show first 3
        print(f"  ├── {sample['directory']}/")
        print(f"  │   └── inputs_embeds.npy    # {sample['shape']}")
    if len(metadata["samples"]) > 3:
        print(f"  ├── ... ({len(metadata['samples']) - 3} more samples)")
    print(f"  ├── metadata.json")
    print(f"  └── npy_files.txt")
    print(f"\nUsage:")
    print(f"  # Load a sample")
    print(f"  inputs_embeds = np.load('{output_dir}/sample_000/inputs_embeds.npy')")
    print(f"  # Or load all paths from the list")
    print(f"  with open('{output_dir}/npy_files.txt', 'r') as f:")
    print(f"      npy_paths = [line.strip() for line in f]")

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate calibration data for Qwen2-VL language model quantization"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./calibration_data/language",
        help="Directory to save calibration data",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Target image size (height width)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of calibration samples (default: all available)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=500,
        help="Maximum tokens to generate per sample",
    )

    args = parser.parse_args()

    generate_language_calibration_data(
        model_name=args.model_name,
        output_dir=args.output_dir,
        image_size=tuple(args.image_size),
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
    )
