import json
import os
import random
import shutil
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from qbcompiler.calibration.utils_calib import list_calib_files_in_json
from qbcompiler.model_dict_legacy.parser.backend.fx_hf_extensions.transformers.models.qwen3vl import (
    repreprocess_pixel_values,
)
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
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

# Order matches mblt-model-zoo's _prepare_dynamic_npu_inputs: [rope, pos, folded].
# Must match the feed_dict order in compile_encoder.py --dynamic.
VISION_DYNAMIC_INPUT_NAMES = ["cos", "pos_embeds/reshape", "float_1_channel_last"]
LANGUAGE_DYNAMIC_INPUT_NAMES = ["inputs_embeds", "deepstack_visual_embeds", "cos"]


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


def pack_rotate_tensor(cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Pack (cos, sin) into the NPU rotateTensor interleave layout.

    Each frequency pair emits a 2x2 rotation block flattened as
    ``[cos, -sin, ..., sin, cos]``. The output width is ``2 * dim`` (no
    alignment padding) which is the calibration-side contract; the runtime
    pads each half to 64-channel PE granularity separately.
    """
    if cos.shape != sin.shape:
        raise ValueError(f"cos/sin shape mismatch: {tuple(cos.shape)} vs {tuple(sin.shape)}")
    dim = cos.shape[-1]
    if dim % 2:
        raise ValueError(f"last axis must be even, got {dim}")
    half = dim // 2
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)
    out = torch.zeros(*cos.shape[:-1], 2 * dim, dtype=torch.float32, device=cos.device)
    out[..., 0 : dim : 2] = cos[..., :half]
    out[..., 1 : dim : 2] = -sin[..., :half]
    out[..., dim : 2 * dim : 2] = sin[..., half:dim]
    out[..., dim + 1 : 2 * dim : 2] = cos[..., half:dim]
    return out


def fold_pixel_values(pixel_values: torch.Tensor) -> torch.Tensor:
    """Fold processor pixel_values ``(N, fold_in)`` into ``(1, fold_in, 1, N)``.

    The dynamic vision encoder consumes this layout: the size-varying axis N
    lives at the trailing position and the patch-content axis becomes the
    conv-channel axis, so the graph is size-agnostic.
    """
    n, fold_in = pixel_values.shape
    return pixel_values.transpose(0, 1).reshape(1, fold_in, 1, n).contiguous()


def save_language_sample_static(sample_dir: Path, inputs_embeds: np.ndarray, deepstack: np.ndarray) -> None:
    sample_dir.mkdir(parents=True)
    np.save(sample_dir / "inputs_embeds.npy", inputs_embeds)
    np.save(sample_dir / "deepstack_visual_embeds.npy", deepstack)


def save_language_sample_dynamic(
    sample_dir: Path,
    inputs_embeds: np.ndarray,
    deepstack: np.ndarray,
    cos: np.ndarray,
) -> None:
    sample_dir.mkdir(parents=True)
    np.save(sample_dir / "inputs_embeds.npy", inputs_embeds)
    np.save(sample_dir / "deepstack_visual_embeds.npy", deepstack)
    np.save(sample_dir / "cos.npy", cos)


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


def compute_language_rope(
    rotary_emb,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
) -> np.ndarray:
    """Return the packed cos/sin rope tensor for a language sample.

    Shape: ``(1, S, 2 * head_dim)``. transformers 4.57.x mrope position_ids
    carry a leading aggregate axis so we trim to the first three (t/h/w) that
    the rotary embedding expects.
    """
    if position_ids.shape[0] > 3:
        position_ids = position_ids[:3]
    with torch.inference_mode():
        cos, sin = rotary_emb(inputs_embeds, position_ids.to(inputs_embeds.device))
    return pack_rotate_tensor(cos, sin).cpu().numpy()


def create_language_manifest_static(stage_dir: Path, hidden_size: int) -> None:
    list_calib_files_in_json(
        str(stage_dir),
        str(stage_dir / "npy_files.json"),
        input_names=["inputs_embeds", "deepstack_visual_embeds"],
        input_shapes=[[1, -1, hidden_size], [3, -1, hidden_size]],
    )


def create_language_manifest_dynamic(stage_dir: Path, hidden_size: int, rope_width: int) -> None:
    list_calib_files_in_json(
        str(stage_dir),
        str(stage_dir / "npy_files.json"),
        input_names=LANGUAGE_DYNAMIC_INPUT_NAMES,
        input_shapes=[[1, -1, hidden_size], [3, -1, hidden_size], [1, -1, rope_width]],
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


def generate_batch(
    model,
    processor,
    image_paths,
    prompts,
    image_size,
    max_new_tokens: int,
    dynamic: bool,
):
    images = []
    texts = []
    for image_path, prompt in zip(image_paths, prompts):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if not dynamic and image_size is not None and image.size != image_size:
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
    # Dynamic mode also needs position_ids so we can compute the rope tensor
    # for each captured sample; static mode ignores it.
    capture_names = ("inputs_embeds", "deepstack_visual_embeds", "visual_pos_masks", "position_ids")

    def capture_language_inputs(_module, _args, kwargs):
        if captured:
            return
        for name in capture_names:
            if kwargs.get(name) is not None:
                captured[name] = move_to_cpu(kwargs[name])

    with model.model.language_model.register_forward_pre_hook(capture_language_inputs, with_kwargs=True):
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    if "inputs_embeds" not in captured:
        raise RuntimeError(f"Language input capture returned no data for {image_paths[0]}")
    return inputs, generated, captured


def save_vision_sample_static(
    vision_dir: Path,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
) -> Path:
    images = repreprocess_pixel_values(pixel_values.float(), grid_thw)
    image_array = images.squeeze(0).permute(1, 2, 0).cpu().numpy()
    vision_dir.mkdir()
    path = vision_dir / "images.npy"
    np.save(path, image_array)
    return path


def save_vision_sample_dynamic(
    vision_dir: Path,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    visual_module,
) -> list[Path]:
    grid_thw_2d = grid_thw.unsqueeze(0) if grid_thw.dim() == 1 else grid_thw
    n = int(torch.prod(grid_thw_2d[0]).item())

    folded = fold_pixel_values(pixel_values.float())
    folded_cl = folded.permute(0, 2, 3, 1).contiguous().to(torch.float32)

    with torch.no_grad():
        pos_embeds = visual_module.fast_pos_embed_interpolate(grid_thw_2d.to(pixel_values.device))
        rotary = visual_module.rot_pos_emb(grid_thw_2d.to(pixel_values.device))
    pos_cl = pos_embeds.reshape(1, 1, n, -1).to(torch.float32)
    emb = torch.cat((rotary, rotary), dim=-1)
    rt = pack_rotate_tensor(emb.cos(), emb.sin())
    rt_cl = rt.reshape(1, 1, n, -1)

    vision_dir.mkdir()
    paths = []
    for key, tensor in (
        ("cos", rt_cl),
        ("pos_embeds", pos_cl),
        ("float_1_channel_last", folded_cl),
    ):
        path = vision_dir / f"{key}.npy"
        np.save(path, tensor.cpu().numpy())
        paths.append(path)
    return paths


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
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Emit 3-input vision samples (folded pixel values, pos_embeds, packed rope) and "
        "3-input decoder samples (inputs_embeds, deepstack, packed rope) for the dynamic MXQ "
        "contract. Requires images downloaded with download_images.py --dynamic.",
    )
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
        args.model_id,
        dtype=torch.float32,
        device_map=args.device,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "left"

    directories = {name: args.output_dir / name for name in ("vision", "prefill", "decode", "language")}
    for name in ("vision", "prefill", "decode"):
        directories[name].mkdir(parents=True)

    ratios = sorted(set((*args.intermediate_ratios, 1.0)))
    embedding_layer = model.model.language_model.embed_tokens
    hidden_size = model.config.text_config.hidden_size
    eos_token_ids = model.generation_config.eos_token_id
    text_head_dim = getattr(model.config.text_config, "head_dim", None) or (
        hidden_size // model.config.text_config.num_attention_heads
    )
    language_rope_width = 2 * text_head_dim
    counts = {"vision": 0, "prefill": 0, "decode": 0}
    vision_calib_paths: list[list[str]] = []
    vision_single_paths: list[str] = []
    visual_module = model.model.visual if args.dynamic else None
    rotary_emb = model.model.language_model.rotary_emb if args.dynamic else None

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
            tuple(args.image_size) if args.image_size else None,
            args.max_new_tokens,
            args.dynamic,
        )

        deepstack = captured.get("deepstack_visual_embeds")
        visual_masks = captured.get("visual_pos_masks")
        if not isinstance(deepstack, list) or len(deepstack) != 3:
            raise RuntimeError(f"Expected three DeepStack tensors for {batch_paths[0]}")
        if visual_masks is None:
            raise RuntimeError(f"visual_pos_masks is missing for {batch_paths[0]}")
        if args.dynamic and captured.get("position_ids") is None:
            raise RuntimeError(f"position_ids capture failed for {batch_paths[0]}")

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

            vision_dir = directories["vision"] / f"sample_{counts['vision']:03d}"
            if args.dynamic:
                paths = save_vision_sample_dynamic(vision_dir, pixel_values, grid_thw, visual_module)
                vision_calib_paths.append([str(path.resolve()) for path in paths])
            else:
                path = save_vision_sample_static(vision_dir, pixel_values, grid_thw)
                vision_single_paths.append(str(path.resolve()))
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

            prefill_dir = directories["prefill"] / f"sample_{counts['prefill']:03d}"
            prefill_embeds_np = prefill_embeddings.numpy()
            deepstack_np = np.concatenate(deepstack_arrays, axis=0)
            if args.dynamic:
                sample_position_ids = captured["position_ids"][:, batch_index : batch_index + 1, real_start:]
                prefill_cos = compute_language_rope(rotary_emb, prefill_embeddings.to(model.device), sample_position_ids)
                save_language_sample_dynamic(prefill_dir, prefill_embeds_np, deepstack_np, prefill_cos)
            else:
                save_language_sample_static(prefill_dir, prefill_embeds_np, deepstack_np)
            counts["prefill"] += 1

            for ratio in ratios:
                token_count = max(1, int(len(decode_ids) * ratio))
                decode_embeddings = tokens_to_embeddings(decode_ids[:token_count], embedding_layer, model.device)
                decode_deepstack = np.zeros((3, token_count, hidden_size), dtype=np.float32)
                decode_dir = directories["decode"] / f"sample_{counts['decode']:03d}"
                if args.dynamic:
                    decode_position_ids = (
                        torch.arange(token_count, dtype=torch.long)
                        .view(1, 1, -1)
                        .expand(3, 1, -1)
                        .contiguous()
                    )
                    decode_embeds_tensor = torch.from_numpy(decode_embeddings).to(model.device)
                    decode_cos = compute_language_rope(rotary_emb, decode_embeds_tensor, decode_position_ids)
                    save_language_sample_dynamic(decode_dir, decode_embeddings, decode_deepstack, decode_cos)
                else:
                    save_language_sample_static(decode_dir, decode_embeddings, decode_deepstack)
                counts["decode"] += 1

        if pixel_offset != inputs["pixel_values"].shape[0]:
            raise RuntimeError("The batched vision inputs were not split completely.")
        if deepstack_offset != deepstack[0].shape[0]:
            raise RuntimeError("The batched DeepStack inputs were not split completely.")

    if not counts["vision"] or not counts["prefill"] or not counts["decode"]:
        raise RuntimeError(f"Insufficient calibration output: {counts}")

    if args.dynamic:
        vision_head_dim = model.config.vision_config.hidden_size // model.config.vision_config.num_heads
        vision_hidden = model.config.vision_config.hidden_size
        fold_in = (
            model.config.vision_config.in_channels
            * model.config.vision_config.temporal_patch_size
            * (model.config.vision_config.patch_size**2)
        )
        vision_index = {
            "info": {
                "input names": VISION_DYNAMIC_INPUT_NAMES,
                "input shapes": [
                    [1, 1, -1, 2 * vision_head_dim],
                    [1, 1, -1, vision_hidden],
                    [1, 1, -1, fold_in],
                ],
            },
            "calib paths": vision_calib_paths,
        }
        (directories["vision"] / "npy_files.json").write_text(
            json.dumps(vision_index, indent=2) + "\n", encoding="utf-8"
        )
        create_language_manifest_dynamic(directories["prefill"], hidden_size, language_rope_width)
        create_language_manifest_dynamic(directories["decode"], hidden_size, language_rope_width)
    else:
        (directories["vision"] / "npy_files.txt").write_text(
            "\n".join(vision_single_paths) + "\n", encoding="utf-8"
        )
        create_language_manifest_static(directories["prefill"], hidden_size)
        create_language_manifest_static(directories["decode"], hidden_size)

    counts["language"] = merge_language_data(directories["prefill"], directories["decode"], directories["language"])
    print(f"Saved calibration data: {counts}")
