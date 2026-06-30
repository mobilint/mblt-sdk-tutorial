"""Calibration data generation for Qwen2-VL (language decoder + vision encoder).

Both calibration sets are produced in a single run from one loaded model. Images
are read from ./images (all *.jpg) and cycled through diverse prompts.

Output:
  calibration_data/language/sample_NNN/inputs_embeds.npy   # [1, seq_len, hidden]
  calibration_data/vision/sample_NNN/images.npy            # [896, 56, 6]
  each target dir also gets metadata.json and npy_files.txt (absolute paths).
"""

import argparse
import glob
import json
import os
import traceback

import numpy as np
import torch
from qbcompiler.model_dict.parser.backend.fx_hf_extensions.transformers.models.qwen2vl import (
    repreprocess_pixel_values,
)
from qbcompiler.model_dict.parser.backend.hf.util import (
    DefaultInputsCaptureContainer,
    InputCaptureCtxManager,
)
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

IMAGE_SIZE = (224, 224)

# (label, prompt) — label is used only in the sample name / metadata.
PROMPT_TEMPLATES = [
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


def load_model_and_processor(model_name):
    """Load Qwen2-VL model and processor from HuggingFace."""
    print(f"Loading model and processor from {model_name}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def build_samples(images_dir="./images"):
    """One sample per image, cycling through PROMPT_TEMPLATES for diversity."""
    image_files = sorted(glob.glob(f"{images_dir}/*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"No JPG images found in {images_dir}")
    samples = []
    for idx, image_path in enumerate(image_files):
        label, prompt = PROMPT_TEMPLATES[idx % len(PROMPT_TEMPLATES)]
        samples.append({"name": f"{label}_{idx:03d}", "image_url": image_path, "prompt": prompt})
    return samples


def _prepare_inputs(processor, sample, model_device):
    """Build chat-template inputs (text + resized image) for one sample."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image_url"]},
                {"type": "text", "text": sample["prompt"]},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    if image_inputs:
        image_inputs = [img.resize(IMAGE_SIZE) for img in image_inputs]
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    return inputs.to(model_device)


def _compute_inputs_embeds(model, captured):
    """Recompute inputs_embeds from input_ids and merge vision features (fallback path)."""
    input_ids = captured.get("input_ids")
    pixel_values = captured.get("pixel_values")
    image_grid_thw = captured.get("image_grid_thw")
    if input_ids is None:
        raise ValueError("Cannot compute inputs_embeds: input_ids not found")

    with torch.no_grad():
        inputs_embeds = model.get_input_embeddings()(input_ids.to(model.device))
        if pixel_values is not None and image_grid_thw is not None:
            image_embeds = model.visual(pixel_values.to(model.device), grid_thw=image_grid_thw.to(model.device))
            n_tokens = (input_ids == model.config.image_token_id).sum().item()
            if n_tokens != image_embeds.shape[0]:
                raise ValueError(f"Image tokens/features mismatch: {n_tokens} vs {image_embeds.shape[0]}")
            mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(mask, image_embeds)
    return inputs_embeds


def capture_language(model, processor, sample, max_new_tokens=500):
    """Capture decoder inputs_embeds (text + merged vision features). Returns {name: ndarray}."""
    inputs = _prepare_inputs(processor, sample, model.device)
    container = DefaultInputsCaptureContainer()
    with InputCaptureCtxManager(model.model, 1, container):
        model.generate(**inputs, max_new_tokens=max_new_tokens)
    captured = container.captured_kwargs[-1]

    embeds = captured.get("inputs_embeds")
    inputs_embeds = embeds if isinstance(embeds, torch.Tensor) else _compute_inputs_embeds(model, captured)
    if inputs_embeds.dtype == torch.bfloat16:
        inputs_embeds = inputs_embeds.float()
    return {"inputs_embeds": inputs_embeds.cpu().numpy()}


def capture_vision(model, processor, sample, max_new_tokens=20):
    """Capture repreprocessed vision pixel values [896, 56, 6]. Returns {name: ndarray}."""
    inputs = _prepare_inputs(processor, sample, model.device)
    container = DefaultInputsCaptureContainer()
    with InputCaptureCtxManager(model.visual, 1, container):
        model.generate(**inputs, max_new_tokens=max_new_tokens)
    pixel_values = container.captured_args[0][0]
    grid_thw = container.captured_kwargs[0].get("grid_thw")[0]
    if pixel_values.dtype == torch.bfloat16:
        pixel_values = pixel_values.float()
    images = repreprocess_pixel_values(pixel_values, grid_thw)  # [gt, 6, H, W]
    return {"images": images[0].permute(1, 2, 0).cpu().numpy()}  # [H, W, 6]


def generate_target(target, capture_fn, model, processor, samples, output_dir, model_name):
    """Run capture_fn over all samples, saving npy files, npy_files.txt and metadata.json."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[{target}] {len(samples)} samples -> {output_dir}")

    npy_paths, meta_samples = [], []
    for i, sample in enumerate(samples):
        try:
            captured = capture_fn(model, processor, sample)
        except Exception as e:
            print(f"  [{i + 1}/{len(samples)}] {sample['name']}: FAILED ({e})")
            traceback.print_exc()
            continue

        sample_dir = os.path.join(output_dir, f"sample_{i:03d}")
        os.makedirs(sample_dir, exist_ok=True)
        shapes = {}
        for key, value in captured.items():
            npy_path = os.path.join(sample_dir, f"{key}.npy")
            np.save(npy_path, value)
            npy_paths.append(os.path.abspath(npy_path))
            shapes[key] = list(value.shape)
        meta_samples.append({"index": i, "name": sample["name"], "prompt": sample["prompt"],
                             "image_url": sample["image_url"], "directory": f"sample_{i:03d}", "shapes": shapes})
        print(f"  [{i + 1}/{len(samples)}] {sample['name']}: {shapes}")

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump({"model_name": model_name, "target": target, "image_size": list(IMAGE_SIZE),
                   "num_samples": len(meta_samples), "samples": meta_samples}, f, indent=2)
    with open(os.path.join(output_dir, "npy_files.txt"), "w") as f:
        f.write("\n".join(npy_paths) + "\n")
    print(f"[{target}] done: {len(meta_samples)} samples, {len(npy_paths)} npy files")


def main():
    parser = argparse.ArgumentParser(description="Generate Qwen2-VL calibration data (language + vision)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="HuggingFace model id")
    parser.add_argument("--output-dir", type=str, default="./calibration_data", help="Base output directory")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit number of samples (default: all images)")
    parser.add_argument("--max-new-tokens", type=int, default=500, help="Max tokens for the language capture pass")
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model_name)
    samples = build_samples()
    if args.num_samples is not None:
        samples = samples[: args.num_samples]

    generate_target("language", lambda m, p, s: capture_language(m, p, s, args.max_new_tokens),
                     model, processor, samples, os.path.join(args.output_dir, "language"), args.model_name)
    generate_target("vision", capture_vision,
                     model, processor, samples, os.path.join(args.output_dir, "vision"), args.model_name)


if __name__ == "__main__":
    main()
