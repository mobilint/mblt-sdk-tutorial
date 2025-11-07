import os
from argparse import ArgumentParser
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import numpy as np

tokenizer_dir = "./Llama-3.2-1B-Instruct"
embedding_dir = "./Llama-3.2-1B-Instruct_embedding_weight.pt"
calib_dir = "./"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=["en", "ko", "ja", "zh", "fr", "de", "es", "ru", "hi", "pt"],
    )
    parser.add_argument(
        "--min_seqlen",
        type=int,
        default=512,
        help="sequence length minimum",
    )
    parser.add_argument(
        "--max_seqlen",
        type=int,
        default=2048,
        help="sequence length maximum",
    )
    parser.add_argument(
        "--max_calib",
        type=int,
        default=128,
        help="maximum number of calibration samples",
    )

    args = parser.parse_args()

    lang = args.lang
    min_seqlen = args.min_seqlen  # sequence length minimum
    max_seqlen = args.max_seqlen  # sequence length maximum
    max_calib = args.max_calib  # number of calibration samples

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=True,
    )  # tokenizer for model

    model_name = tokenizer_dir.split("/")[-1]

    dataset = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train")[
        "text"
    ]

    embedding_weight = torch.load(embedding_dir)  # (vocab_size, dim) embedding weight
    embedding_layer = torch.nn.Embedding.from_pretrained(embedding_weight).to(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )  # set embedding layer

    os.makedirs(
        os.path.join(calib_dir, f"{model_name}-Wikipedia-{lang}"), exist_ok=True
    )  # calibration save dir

    calib_count = 0  # current number of calibration samples
    with open(os.path.join(calib_dir, f"{model_name}-Wikipedia-{lang}.txt"), "w") as f:
        f.write("")
    for i, text in enumerate(dataset):
        print(f"Sentence {i}")
        token_ids = (
            tokenizer(text, return_tensors="pt")["input_ids"]
            .squeeze()
            .to("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(token_ids.shape)
        if token_ids.ndim == 0:  # empty
            continue
        embedded_text = embedding_layer(token_ids)
        if embedded_text.ndim == 1:
            embedded_text = embedded_text.unsqueeze(0).unsqueeze(0)
        elif embedded_text.ndim == 2:
            embedded_text = embedded_text.unsqueeze(0)
        print(embedded_text.shape)
        if embedded_text.shape[1] < min_seqlen:
            continue
        elif embedded_text.shape[1] > max_seqlen:
            embedded_text = embedded_text[:, :max_seqlen, :]

        np.save(
            os.path.join(
                calib_dir,
                f"{model_name}-Wikipedia-{lang}/inputs_embeds_{calib_count}.npy",
            ),
            embedded_text.to(device="cpu", dtype=torch.float32).numpy(),
        )
        with open(
            os.path.join(calib_dir, f"{model_name}-Wikipedia-{lang}.txt"), "a"
        ) as f:
            f.write(
                os.path.abspath(
                    os.path.join(
                        calib_dir,
                        f"{model_name}-Wikipedia-{lang}/inputs_embeds_{calib_count}.npy",
                    )
                    + "\n"
                )
            )

        calib_count += 1
        if calib_count >= max_calib:
            break

# Note: The downloaded dataset may continue to occupy disk space.
# You can remove it by running: hf cache delete
