"""Calibration data generation for Qwen3-VL (language decoder + vision encoder).

Qwen3-VL adds a deepstack visual pathway, so the language calibration set carries
both inputs_embeds and deepstack_visual_embeds. Two passes per image are run from
one loaded model:
  Pass 1: capture the language prefill inputs (inputs_embeds, deepstack_visual_embeds).
  Pass 2: full generate to obtain the decode token sequence (EOS-terminated only).
Prefill and decode samples are then merged into a single language/ directory that
qbcompiler consumes through npy_files.json. Vision samples come directly from the
processor's pixel_values / image_grid_thw (no hook needed).

Images are read from ./images (all *.jpg) and cycled through diverse prompts.
Model dimensions (hidden_size, etc.) are detected at runtime, so 2B/4B/8B all
run with the same command.

Output:
  calibration_data/vision/sample_NNN/images.npy               # [H, W, 6]
  calibration_data/vision/npy_files.txt
  calibration_data/prefill/sample_NNN/{inputs_embeds,deepstack_visual_embeds}.npy
  calibration_data/decode/sample_NNN/{inputs_embeds,deepstack_visual_embeds}.npy
  calibration_data/language/{prefill_NNN,decode_NNN}/ + npy_files.json   # merged
"""

import argparse
import glob
import json
import os
import shutil

import numpy as np
import torch
from einops import rearrange
from qbcompiler.calibration.utils_calib import list_calib_files_in_json
from qbcompiler.model_dict.parser.backend.hf.util import (
    DefaultInputsCaptureContainer,
    InputCaptureCtxManager,
)
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

IMAGE_SIZE = (224, 224)

PROMPTS = [
    "Describe this image.",
    "Describe this image in detail, including all objects, colors, textures, and spatial relationships.",
    "What do you see in this image? Provide a thorough description.",
    "What objects can you identify in this image?",
    "List all objects you can see in this image.",
    "Count and list all distinct objects or elements you can identify in this image.",
    "Describe the scene, setting, and context shown in this image.",
    "What kind of place or setting does this image show?",
    "Analyze what is happening in this image and explain your reasoning.",
    "What story or narrative does this image convey?",
    "Describe the spatial arrangement and positioning of elements in this image.",
    "Analyze the composition and visual structure of this image.",
    "Provide a technical description of what is shown, including materials, structure, and design.",
    "What materials or substances can you identify in this image?",
    "Describe the colors, textures, and visual patterns present in this image.",
    "Describe the lighting, shadows, and overall atmosphere of this image.",
    "What is the purpose or function of the main subject in this image?",
    "What action or activity, if any, is taking place in this image?",
    "Describe any text, symbols, or signage visible in this image.",
    "What small or easily overlooked details can you spot in this image?",
]


def repreprocess_pixel_values(pixel_values, grid_thw):
    """Rearrange the processor's flat patch tensor into the NPU input layout.

    pixel_values: (gt*gh*gw*Mh*Mw, c*pt*ph*pw). grid_thw: (gt, gh, gw). ph=pw (patch size)
    are derived from the tensor at runtime. Example (Qwen3-VL, 224x224): grid (1, 14, 14)
    -> returns (1, 6, 784, 64).
    """
    gt, gh, gw = grid_thw
    c, pt, Mh, Mw = 3, 2, 2, 2
    ph = pw = int((pixel_values.shape[-1] // (pt * c)) ** 0.5)
    images = rearrange(
        pixel_values,
        "(gt gh gw Mh Mw) (c pt ph pw) -> gt (pt c) (gh gw ph) (Mh Mw pw)",
        gt=gt, gh=gh // Mh, gw=gw // Mw, Mh=Mh, Mw=Mw, c=c, pt=pt, ph=ph, pw=pw,
    )
    return images


def build_samples(images_dir="./images"):
    """One sample per image, cycling through PROMPTS for diversity."""
    image_files = sorted(glob.glob(f"{images_dir}/*.jpg"))
    if not image_files:
        raise FileNotFoundError(f"No JPG images found in {images_dir}")
    samples = []
    for idx, image_path in enumerate(image_files):
        samples.append({"image_url": image_path, "prompt": PROMPTS[idx % len(PROMPTS)]})
    return samples


def save_lang_sample(sample_dir, inputs_embeds_np, deepstack_np):
    """Save one language calibration sample (inputs_embeds + deepstack_visual_embeds)."""
    os.makedirs(sample_dir, exist_ok=True)
    np.save(os.path.join(sample_dir, "inputs_embeds.npy"), inputs_embeds_np)
    np.save(os.path.join(sample_dir, "deepstack_visual_embeds.npy"), deepstack_np)


def merge_language_dirs(input_dirs, output_dir):
    """Merge N language npy_files.json directories into one (seq_len axis set to -1)."""
    all_data, dir_labels = [], []
    for input_dir in input_dirs:
        json_path = os.path.join(input_dir, "npy_files.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Not found: {json_path}")
        with open(json_path) as f:
            all_data.append(json.load(f))
        dir_labels.append(os.path.basename(input_dir.rstrip("/")))

    ref_info = all_data[0]["info"]
    os.makedirs(output_dir, exist_ok=True)
    merged_calib_paths = []
    for data, label in zip(all_data, dir_labels):
        for i, path_pair in enumerate(data["calib paths"]):
            src_dir = os.path.dirname(path_pair[0])
            dst_dir = os.path.join(output_dir, f"{label}_{i:03d}")
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            merged_calib_paths.append(
                [os.path.abspath(os.path.join(dst_dir, os.path.basename(p))) for p in path_pair]
            )

    merged_shapes = [list(s) for s in ref_info["input shapes"]]
    for s in merged_shapes:
        s[1] = -1
    merged_data = {
        "info": {"input names": ref_info["input names"], "input shapes": merged_shapes},
        "calib paths": merged_calib_paths,
    }
    with open(os.path.join(output_dir, "npy_files.json"), "w") as f:
        json.dump(merged_data, f, indent=4)
    return len(merged_calib_paths)


def generate(model, processor, samples, output_dir, max_new_tokens, intermediate_ratios):
    """Produce vision + prefill + decode samples, then merge into language/."""
    dirs = {name: os.path.join(output_dir, name) for name in ("vision", "prefill", "decode", "language")}
    for name in ("vision", "prefill", "decode"):
        os.makedirs(dirs[name], exist_ok=True)

    embed_tokens = model.model.language_model.embed_tokens
    eos_token_id = processor.tokenizer.eos_token_id
    pad_token_id = processor.tokenizer.pad_token_id

    ratios = sorted(intermediate_ratios)
    if 1.0 not in ratios:
        ratios.append(1.0)

    counts = {"vision": 0, "prefill": 0, "decode": 0}
    vision_npy_paths = []

    for idx, sample in enumerate(samples):
        print(f"\n[{idx + 1}/{len(samples)}] {os.path.basename(sample['image_url'])}")
        # Feed the loaded image directly to the Qwen3-VL processor.
        image = Image.open(sample["image_url"]).convert("RGB")
        if image.size != IMAGE_SIZE:
            raise ValueError(
                f"{sample['image_url']}: image size {image.size} (W, H) != IMAGE_SIZE "
                f"{IMAGE_SIZE}. Prepare the calibration images at exactly this size first."
            )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": sample["prompt"]},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

        input_len = inputs["input_ids"].shape[1]
        vis_pixel_values = inputs["pixel_values"].float()
        vis_grid_thw = inputs["image_grid_thw"]

        # Pass 1: capture language prefill inputs.
        lang_container = DefaultInputsCaptureContainer()
        try:
            with InputCaptureCtxManager(model.model.language_model, 1, lang_container):
                model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        except Exception as e:
            print(f"  prefill capture failed: {e}")
            continue
        if not lang_container.captured_kwargs:
            print("  WARNING: language hook captured nothing, skipping")
            continue

        # Pass 2: full generate for the decode sequence.
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        lang_kwargs = lang_container.captured_kwargs[-1]
        prefill_embeds = lang_kwargs["inputs_embeds"].float()
        hidden_size = prefill_embeds.shape[2]
        deepstack = lang_kwargs.get("deepstack_visual_embeds")
        vis_masks = lang_kwargs.get("visual_pos_masks")

        output_ids = generated[0].tolist()[input_len:]
        if pad_token_id is not None:
            output_ids = [t for t in output_ids if t != pad_token_id]
        if not output_ids or eos_token_id not in output_ids:
            print("  WARNING: EOS not generated, skipping")
            continue
        decode_token_ids = output_ids[:output_ids.index(eos_token_id)]
        if not decode_token_ids:
            print("  WARNING: no tokens before EOS, skipping")
            continue
        n_total = len(decode_token_ids)

        # ===== Vision =====
        images = repreprocess_pixel_values(vis_pixel_values, vis_grid_thw[0])
        images_np = images.squeeze(0).permute(1, 2, 0).cpu().numpy()
        sample_dir = os.path.join(dirs["vision"], f"sample_{counts['vision']:03d}")
        os.makedirs(sample_dir, exist_ok=True)
        npy_path = os.path.join(sample_dir, "images.npy")
        np.save(npy_path, images_np)
        vision_npy_paths.append(os.path.abspath(npy_path))
        counts["vision"] += 1

        # ===== Prefill =====
        prefill_embeds_np = prefill_embeds.cpu().numpy()
        seq_len = prefill_embeds_np.shape[1]
        if (deepstack is not None and isinstance(deepstack, list)
                and len(deepstack) == 3 and vis_masks is not None):
            vis_mask = vis_masks[0]
            ds_padded = []
            for ds_tensor in deepstack:
                ds_full = ds_tensor.float()
                padded = torch.zeros(1, seq_len, hidden_size, dtype=ds_full.dtype, device=ds_full.device)
                padded[0, vis_mask, :] = ds_full
                ds_padded.append(padded.cpu().numpy())
            prefill_deepstack_np = np.concatenate(ds_padded, axis=0)
        else:
            print("  WARNING: no deepstack found, using zeros for prefill")
            prefill_deepstack_np = np.zeros((3, seq_len, hidden_size), dtype=np.float32)
        save_lang_sample(os.path.join(dirs["prefill"], f"sample_{counts['prefill']:03d}"),
                         prefill_embeds_np, prefill_deepstack_np)
        counts["prefill"] += 1

        # ===== Decode (intermediate ratios + full sequence up to EOS) =====
        for ratio in ratios:
            n_tokens = max(1, int(n_total * ratio))
            prefix_ids = torch.tensor([decode_token_ids[:n_tokens]], dtype=torch.long, device=model.device)
            with torch.no_grad():
                prefix_embeds_np = embed_tokens(prefix_ids).float().cpu().numpy()
            prefix_deepstack_np = np.zeros((3, prefix_embeds_np.shape[1], hidden_size), dtype=np.float32)
            save_lang_sample(os.path.join(dirs["decode"], f"sample_{counts['decode']:03d}"),
                             prefix_embeds_np, prefix_deepstack_np)
            counts["decode"] += 1

    # ===== Index files =====
    with open(os.path.join(dirs["vision"], "npy_files.txt"), "w") as f:
        f.write("\n".join(vision_npy_paths) + "\n")
    for stage in ("prefill", "decode"):
        if counts[stage] > 0:
            list_calib_files_in_json(dirs[stage], os.path.join(dirs[stage], "npy_files.json"))

    # ===== Merge prefill + decode into language/ =====
    if counts["prefill"] > 0 and counts["decode"] > 0:
        merge_language_dirs([dirs["prefill"], dirs["decode"]], dirs["language"])
    else:
        print(f"WARNING: prefill={counts['prefill']}, decode={counts['decode']}, skipping merge")

    print(f"Done: vision={counts['vision']}, language={counts['prefill'] + counts['decode']} samples")


def main():
    parser = argparse.ArgumentParser(description="Generate Qwen3-VL calibration data (language + vision)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-4B-Instruct", help="HuggingFace model id")
    parser.add_argument("--output-dir", type=str, default="./calibration_data", help="Base output directory")
    parser.add_argument("--num-samples", type=int, default=100, help="Limit number of samples (default: all images)")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max tokens for the decode generation pass")
    parser.add_argument("--intermediate-ratios", type=float, nargs="*", default=[0.25, 0.5, 0.75],
                        help="Decode-prefix ratios to save (1.0 is always appended)")
    args = parser.parse_args()

    print(f"Loading model and processor from {args.model_name}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name, dtype=torch.float32, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model_name)
    processor.tokenizer.padding_side = "left"

    samples = build_samples()
    if args.num_samples is not None:
        samples = samples[: args.num_samples]

    generate(model, processor, samples, args.output_dir, args.max_new_tokens, args.intermediate_ratios)


if __name__ == "__main__":
    main()
