import json
import os
import random
import shutil
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from compile_config import DECODER_INPUT_NAMES
from PIL import Image
from qbcompiler.calibration.utils_calib import list_calib_files_in_json
from qbcompiler.model_dict_new.parser.patcher.models.hf_models.qwen3vl import fold_pixel_values
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
SEED = 42
PROMPTS = (
    "Describe this image.",
    "Describe this image in detail, including objects, colors, textures, and spatial relationships.",
    "What objects can you identify in this image?",
    "List all distinct objects visible in this image.",
    "Describe the scene, setting, and context shown in this image.",
    "Analyze what is happening in this image and explain your reasoning.",
    "What story or narrative does this image convey?",
    "Describe how the elements in this image are arranged relative to each other.",
    "Describe the lighting, shadows, colors, and overall atmosphere.",
    "What small or easily overlooked details can you spot?",
)


def set_seed() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def save_language_sample(sample_dir: Path, inputs_embeds: np.ndarray, deepstack: Sequence[np.ndarray]) -> None:
    sample_dir.mkdir(parents=True)
    np.save(sample_dir / "inputs_embeds.npy", inputs_embeds)
    for index, tensor in enumerate(deepstack):
        np.save(sample_dir / f"deepstack_visual_embeds_{index}.npy", tensor)


def tokens_to_embeddings(token_ids: Sequence[int], embedding_layer, device) -> np.ndarray:
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        return embedding_layer(tokens).float().cpu().numpy()


def tokens_before_eos(token_ids: Sequence[int], eos_token_ids: int | Sequence[int]) -> list[int] | None:
    eos_ids = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids)
    eos_positions = [index for index, token_id in enumerate(token_ids) if token_id in eos_ids]
    if not eos_positions:
        return None
    return list(token_ids[: eos_positions[0]])


def create_language_manifest(stage_dir: Path, hidden_size: int) -> None:
    list_calib_files_in_json(
        str(stage_dir),
        str(stage_dir / "npy_files.json"),
        input_names=DECODER_INPUT_NAMES,
        input_shapes=[[1, -1, hidden_size]] * len(DECODER_INPUT_NAMES),
    )


def merge_language_data(prefill_dir: Path, decode_dir: Path, output_dir: Path) -> int:
    manifests = []
    for source_dir in (prefill_dir, decode_dir):
        manifest_path = source_dir / "npy_files.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Calibration manifest not found: {manifest_path}")
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))

    reference = manifests[0]["info"]
    for manifest in manifests[1:]:
        if manifest["info"] != reference:
            raise ValueError("Prefill and decode calibration manifests have different input contracts.")

    output_dir.mkdir(parents=True)
    merged_paths = []
    for source_dir, manifest in zip((prefill_dir, decode_dir), manifests):
        for index, source_paths in enumerate(manifest["calib paths"]):
            source_sample = Path(source_paths[0]).parent
            destination = output_dir / f"{source_dir.name}_{index:03d}"
            shutil.copytree(source_sample, destination)
            merged_paths.append([str((destination / Path(path).name).resolve()) for path in source_paths])

    output = {"info": reference, "calib paths": merged_paths}
    (output_dir / "npy_files.json").write_text(json.dumps(output, indent=4) + "\n", encoding="utf-8")
    return len(merged_paths)


def move_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [move_to_cpu(item) for item in value]
    return value


def generate_batch(model, processor, image_paths, prompts, image_size, max_new_tokens: int):
    images = []
    texts = []
    for image_path, prompt in zip(image_paths, prompts):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if image.size != image_size:
            raise ValueError(f"{image_path}: expected image size {image_size}, got {image.size}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        images.append(image)
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
    captured = {}

    def capture_language_inputs(_module, _args, kwargs):
        if captured:
            return
        for name in ("inputs_embeds", "deepstack_visual_embeds", "visual_pos_masks"):
            if kwargs.get(name) is not None:
                captured[name] = move_to_cpu(kwargs[name])

    with model.model.language_model.register_forward_pre_hook(capture_language_inputs, with_kwargs=True):
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    if "inputs_embeds" not in captured:
        raise RuntimeError(f"Language input capture returned no data for {image_paths[0]}")
    return inputs, generated, captured


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate Qwen3-VL encoder and decoder calibration data")
    parser.add_argument("--image-dir", type=Path, default=Path("images"))
    parser.add_argument("--output-dir", type=Path, default=Path("calibration_data"))
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--intermediate-ratios", type=float, nargs="*", default=(0.25, 0.5, 0.75))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if any(ratio <= 0 or ratio > 1 for ratio in args.intermediate_ratios):
        raise ValueError("--intermediate-ratios must be greater than 0 and no greater than 1")
    if args.output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{args.output_dir} already exists. Use --force to replace it.")
        shutil.rmtree(args.output_dir)

    image_files = sorted(args.image_dir.glob("*.jpg"))
    if len(image_files) < args.num_samples:
        raise RuntimeError(f"Found {len(image_files)} images, but --num-samples requests {args.num_samples}")
    image_files = image_files[: args.num_samples]

    set_seed()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        device_map=args.device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"

    directories = {name: args.output_dir / name for name in ("vision", "prefill", "decode", "language")}
    for name in ("vision", "prefill", "decode"):
        directories[name].mkdir(parents=True)

    ratios = sorted(set((*args.intermediate_ratios, 1.0)))
    embedding_layer = model.model.language_model.embed_tokens
    hidden_size = model.config.text_config.hidden_size
    eos_token_ids = model.generation_config.eos_token_id
    counts = {"vision": 0, "prefill": 0, "decode": 0}
    vision_paths = []

    for batch_start in range(0, len(image_files), args.batch_size):
        batch_paths = image_files[batch_start : batch_start + args.batch_size]
        batch_prompts = [PROMPTS[index % len(PROMPTS)] for index in range(batch_start, batch_start + len(batch_paths))]
        batch_end = batch_start + len(batch_paths)
        print(f"[{batch_start + 1}-{batch_end}/{len(image_files)}] {', '.join(path.name for path in batch_paths)}")
        inputs, generated, captured = generate_batch(
            model,
            processor,
            batch_paths,
            batch_prompts,
            tuple(args.image_size),
            args.max_new_tokens,
        )

        deepstack = captured.get("deepstack_visual_embeds")
        visual_masks = captured.get("visual_pos_masks")
        if not isinstance(deepstack, list) or len(deepstack) != 3:
            raise RuntimeError(f"Expected three DeepStack tensors for {batch_paths[0]}")
        if visual_masks is None:
            raise RuntimeError(f"visual_pos_masks is missing for {batch_paths[0]}")

        input_width = inputs["input_ids"].shape[1]
        pixel_offset = 0
        deepstack_offset = 0
        for batch_index, image_path in enumerate(batch_paths):
            grid_thw = inputs["image_grid_thw"][batch_index]
            patch_count = int(torch.prod(grid_thw).item())
            pixel_values = inputs["pixel_values"][pixel_offset : pixel_offset + patch_count]
            pixel_offset += patch_count

            attention_mask = inputs["attention_mask"][batch_index]
            real_start = int((attention_mask == 0).sum().item())
            prefill_embeddings = captured["inputs_embeds"][batch_index : batch_index + 1, real_start:].float()
            visual_mask = visual_masks[batch_index, real_start:].bool()
            visual_count = int(visual_mask.sum().item())
            sample_deepstack = [
                tensor[deepstack_offset : deepstack_offset + visual_count].float() for tensor in deepstack
            ]
            deepstack_offset += visual_count

            output_ids = generated[batch_index].tolist()[input_width:]
            decode_ids = tokens_before_eos(output_ids, eos_token_ids)
            if decode_ids is None:
                print(f"  {image_path.name}: skipped because EOS token was not generated")
                continue
            if not decode_ids:
                print(f"  {image_path.name}: skipped because no token was generated before EOS")
                continue

            images = fold_pixel_values(pixel_values.float())
            image_array = images.squeeze(0).permute(1, 2, 0).cpu().numpy()
            vision_dir = directories["vision"] / f"sample_{counts['vision']:03d}"
            vision_dir.mkdir()
            vision_path = vision_dir / "images.npy"
            np.save(vision_path, image_array)
            vision_paths.append(str(vision_path.resolve()))
            counts["vision"] += 1

            sequence_length = prefill_embeddings.shape[1]
            deepstack_arrays = []
            for tensor in sample_deepstack:
                expected_shape = (visual_count, hidden_size)
                if tuple(tensor.shape) != expected_shape:
                    raise RuntimeError(f"DeepStack shape {tuple(tensor.shape)} does not match {expected_shape}")
                padded = torch.zeros(1, sequence_length, hidden_size, dtype=tensor.dtype)
                padded[0, visual_mask, :] = tensor
                deepstack_arrays.append(padded.numpy())

            save_language_sample(
                directories["prefill"] / f"sample_{counts['prefill']:03d}",
                prefill_embeddings.numpy(),
                deepstack_arrays,
            )
            counts["prefill"] += 1

            for ratio in ratios:
                token_count = max(1, int(len(decode_ids) * ratio))
                decode_embeddings = tokens_to_embeddings(decode_ids[:token_count], embedding_layer, model.device)
                decode_deepstack = [np.zeros((1, token_count, hidden_size), dtype=np.float32) for _ in range(3)]
                save_language_sample(
                    directories["decode"] / f"sample_{counts['decode']:03d}",
                    decode_embeddings,
                    decode_deepstack,
                )
                counts["decode"] += 1

        if pixel_offset != inputs["pixel_values"].shape[0]:
            raise RuntimeError("The batched vision inputs were not split completely.")
        if deepstack_offset != deepstack[0].shape[0]:
            raise RuntimeError("The batched DeepStack inputs were not split completely.")

    if not counts["vision"] or not counts["prefill"] or not counts["decode"]:
        raise RuntimeError(f"Insufficient calibration output: {counts}")

    (directories["vision"] / "npy_files.txt").write_text("\n".join(vision_paths) + "\n", encoding="utf-8")
    create_language_manifest(directories["prefill"], hidden_size)
    create_language_manifest(directories["decode"], hidden_size)
    counts["language"] = merge_language_data(directories["prefill"], directories["decode"], directories["language"])
    print(f"Saved calibration data: {counts}")
