import os
from argparse import ArgumentParser

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SUPPORTED_LANGUAGES = ("en", "de", "fr", "it", "pt", "hi", "es", "th")


def generate_calibration(
    model_id: str,
    languages: list[str],
    output_dir: str = "./calib",
    min_seqlen: int = 512,
    max_seqlen: int = 2048,
    max_calib: int = 128,
) -> None:
    if not languages:
        raise ValueError("languages must not be empty")
    if len(set(languages)) != len(languages):
        raise ValueError("languages must not contain duplicates")
    if max_calib < len(languages):
        raise ValueError("max_calib must be at least the number of languages")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model from: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    embedding_layer = model.get_input_embeddings().to(device).eval()
    vocab_size, embed_dim = embedding_layer.weight.shape
    del model
    print(f"Embedding shape: ({vocab_size}, {embed_dim})")

    model_name = model_id.replace("/", "-")

    language_group = languages[0] if len(languages) == 1 else "multilingual"
    output_path = f"{output_dir}/datas/{model_name}/{language_group}"
    os.makedirs(output_path, exist_ok=True)

    sample_count = 0
    samples_per_language, remainder = divmod(max_calib, len(languages))
    with torch.inference_mode(), tqdm(total=max_calib, desc="Calibrating") as progress:
        for index, language in enumerate(languages):
            target_count = samples_per_language + (index < remainder)
            dataset = load_dataset(
                "wikimedia/wikipedia",
                f"20231101.{language}",
                split="train",
                streaming=True,
            )
            language_count = 0
            progress.set_description(f"Calibrating ({language})")

            for row in dataset:
                token_ids = tokenizer(row["text"], return_tensors="pt")["input_ids"].squeeze(0).to(device)
                if token_ids.shape[0] < min_seqlen:
                    continue

                inputs_embeds = embedding_layer(token_ids[:max_seqlen]).unsqueeze(0)
                np.save(
                    f"{output_path}/{language}_inputs_embeds_{language_count}.npy",
                    inputs_embeds.float().cpu().numpy(),
                )

                language_count += 1
                sample_count += 1
                progress.update()
                if language_count == target_count:
                    break

            if language_count != target_count:
                raise RuntimeError(f"requested {target_count} {language} samples, wrote {language_count}")

    print(f"Saved {sample_count} calibration samples to {output_path}", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
    )
    parser.add_argument("--output-dir", type=str, default="./calibration_data")
    parser.add_argument("--languages", nargs="+", default=list(SUPPORTED_LANGUAGES))
    parser.add_argument(
        "--min-seqlen",
        type=int,
        default=512,
        help="Minimum sequence length (default: 512)",
    )
    parser.add_argument(
        "--max-seqlen",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--max-calib",
        type=int,
        default=128,
        help="Total number of calibration samples (default: 128)",
    )

    args = parser.parse_args()

    generate_calibration(
        model_id=args.model_id,
        languages=args.languages,
        output_dir=args.output_dir,
        min_seqlen=args.min_seqlen,
        max_seqlen=args.max_seqlen,
        max_calib=args.max_calib,
    )

    os._exit(0)
