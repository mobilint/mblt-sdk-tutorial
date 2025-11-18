"""
Vision Encoder Calibration Data Generation for Quantization

This module generates calibration data for quantizing the Qwen2-VL vision encoder.
Each sample is saved as separate .npy files for easy loading during compilation.

Output Structure:
-----------------
calibration_data/vision/
├── sample_000/
│   └── images.npy           # Reprocessed images for vision model [896, 56, 6]
├── sample_001/
│   └── images.npy
├── ...
├── metadata.json            # Sample information and configuration
└── npy_files.txt            # List of all .npy file paths (absolute paths)
"""

import os
import sys
import json
import torch
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

from qubee.model_dict.parser.backend.hf.util import (
    DefaultInputsCaptureContainer,
    InputCaptureCtxManager,
)
from qubee.model_dict.parser.backend.fx_hf_extensions.transformers.models.qwen2vl import (
    repreprocess_pixel_values,
)


def load_model_and_processor(model_name: str):
    """Load Qwen2-VL model and processor from HuggingFace."""
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
        ("detailed_description", "Describe this image in detail, including all objects, colors, textures, and spatial relationships."),
        ("object_identification", "What objects can you identify in this image?"),
        ("scene_understanding", "Describe the scene, setting, and context shown in this image."),
        ("visual_reasoning", "Analyze what is happening in this image and explain your reasoning."),
        ("counting", "Count and list all distinct objects or elements you can identify in this image."),
        ("spatial_reasoning", "Describe the spatial arrangement and positioning of elements in this image."),
        ("technical_description", "Provide a technical description of what is shown, including materials, structure, and design."),
        ("color_texture", "Describe the colors, textures, and visual patterns present in this image."),
        ("comparison", "Compare and contrast the different elements visible in this image."),
        ("purpose_function", "What is the purpose or function of the main subject in this image?"),
        ("environment_context", "Describe the environment and context surrounding the main subject."),
        ("detailed_analysis", "Provide a comprehensive analysis of this image, covering all observable details and their relationships."),
        ("characteristics", "What are the key characteristics and distinctive features of what is shown?"),
        ("composition", "Analyze the composition and visual structure of this image."),
        ("action_activity", "What action or activity, if any, is taking place in this image?"),
        ("categorization", "What category or type does the main subject of this image belong to?"),
        ("materials", "What materials or substances can you identify in this image?"),
        ("lighting_atmosphere", "Describe the lighting, shadows, and overall atmosphere of this image."),
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

        samples.append({
            "name": sample_name,
            "image_url": image_path,
            "prompt": prompt_text,
        })

    print(f"Created {len(samples)} calibration samples from {len(image_files)} images")
    return samples


def capture_vision_encoder_inputs(
    model,
    processor,
    sample_config: Dict,
) -> Dict[str, np.ndarray]:
    """
    Capture vision encoder inputs for a single sample.

    This captures the exact inputs needed for compile_vision.py:
    - images: Pixel values reshaped to (896, 56, 6)

    Args:
        model: Qwen2VLForConditionalGeneration model
        processor: AutoProcessor for Qwen2-VL
        sample_config: Dictionary with 'image_url' and 'prompt'

    Returns:
        Dictionary containing:
        - images: Pixel values [896, 56, 6]
    """
    # Image size is fixed at 224x224
    image_size = (224, 224)

    # Create messages from sample config
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

    # Capture vision encoder inputs
    inputs_container = DefaultInputsCaptureContainer()
    max_call_limit = 1

    with InputCaptureCtxManager(model.visual, max_call_limit, inputs_container) as f:
        # Run short generation to trigger vision encoder
        _ = model.generate(**inputs, max_new_tokens=20)

    # Extract captured inputs
    pixel_values = inputs_container.captured_args[0][0]

    # Convert to float32 if needed (bfloat16 not supported by numpy)
    if pixel_values.dtype == torch.bfloat16:
        pixel_values = pixel_values.float()

    # Reshape to (896, 56, 6) - removed batch dimension
    # pixel_values shape is typically [num_patches, embedding_dim]
    # We need to reshape it to the desired format
    pixel_values_np = pixel_values.cpu().numpy()

    # Reshape to (896, 56, 6)
    # The total size should match: 896 * 56 * 6 = 301,056
    total_elements = pixel_values_np.size
    target_shape = (896, 56, 6)
    target_size = 896 * 56 * 6

    if total_elements >= target_size:
        # Flatten and take first target_size elements
        images = pixel_values_np.flatten()[:target_size].reshape(target_shape)
    else:
        # Pad if needed
        images = np.zeros(target_shape, dtype=pixel_values_np.dtype)
        flat_pv = pixel_values_np.flatten()
        images.flat[:len(flat_pv)] = flat_pv

    return {
        "images": images,
    }


def generate_vision_calibration_data(
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    output_dir: str = "./calibration_data/vision",
    num_samples: int = None,
) -> str:
    """
    Generate calibration data for vision encoder quantization.

    Image size is fixed at 224x224.

    Args:
        model_name: HuggingFace model identifier
        output_dir: Directory to save calibration data
        num_samples: Number of calibration samples (None = all available)

    Returns:
        Path to calibration data directory
    """
    print("=" * 80)
    print("VISION ENCODER CALIBRATION DATA GENERATION")
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
        "image_size": [224, 224],  # Fixed at 224x224
        "num_samples": len(sample_configs),
        "samples": []
    }

    # List to store all npy file paths
    npy_file_paths = []

    # Process each sample
    for i, sample_config in enumerate(sample_configs):
        print(f"\n[{i+1}/{len(sample_configs)}] Processing: {sample_config['name']}")
        print(f"   Image: {sample_config['image_url']}")
        print(f"   Prompt: {sample_config['prompt']}")

        try:
            # Capture inputs
            captured_data = capture_vision_encoder_inputs(
                model, processor, sample_config
            )

            # Create sample directory
            sample_dir = os.path.join(output_dir, f"sample_{i:03d}")
            os.makedirs(sample_dir, exist_ok=True)

            # Save each tensor as separate .npy file
            for key, value in captured_data.items():
                npy_path = os.path.join(sample_dir, f"{key}.npy")
                np.save(npy_path, value)
                print(f"   ✓ Saved {key}: {value.shape} -> {npy_path}")

                # Add absolute path to list
                npy_file_paths.append(os.path.abspath(npy_path))

            # Add to metadata
            metadata["samples"].append({
                "index": i,
                "name": sample_config["name"],
                "prompt": sample_config["prompt"],
                "image_url": sample_config["image_url"],
                "directory": f"sample_{i:03d}",
                "shapes": {key: list(val.shape) for key, val in captured_data.items()}
            })

        except Exception as e:
            print(f"   ✗ Error processing sample: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Metadata saved to: {metadata_path}")

    # Save txt file listing all npy file paths
    npy_list_path = os.path.join(output_dir, "npy_files.txt")
    with open(npy_list_path, 'w') as f:
        for npy_path in npy_file_paths:
            f.write(npy_path + '\n')

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
    print("VISION CALIBRATION DATA GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - Samples collected: {len(metadata['samples'])}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Total size: {total_size_mb:.2f} MB")
    print(f"  - Image size: 224 x 224 (fixed)")
    print(f"\nStructure:")
    print(f"  {output_dir}/")
    for sample in metadata['samples'][:3]:  # Show first 3
        print(f"  ├── {sample['directory']}/")
        print(f"  │   └── images.npy      # {sample['shapes']['images']}")
    if len(metadata['samples']) > 3:
        print(f"  ├── ... ({len(metadata['samples']) - 3} more samples)")
    print(f"  ├── metadata.json")
    print(f"  └── npy_files.txt")
    print(f"\nUsage:")
    print(f"  # Load a sample")
    print(f"  images = np.load('{output_dir}/sample_000/images.npy')")
    print(f"  # Shape: (896, 56, 6)")
    print(f"  # Or load all paths from the list")
    print(f"  with open('{output_dir}/npy_files.txt', 'r') as f:")
    print(f"      npy_paths = [line.strip() for line in f]")

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate calibration data for Qwen2-VL vision encoder quantization (image size fixed at 224x224)"
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
        default="./calibration_data/vision",
        help="Directory to save calibration data",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of calibration samples (default: all available)",
    )

    args = parser.parse_args()

    generate_vision_calibration_data(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
    )
