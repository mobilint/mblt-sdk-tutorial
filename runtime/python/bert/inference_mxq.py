from argparse import ArgumentParser
from pathlib import Path

import torch
from transformers import BertTokenizer
from wrapper.bert_model import BertMXQ

DUMMY_CORPUS = [
    ["A man is eating food.", "A man is eating something."],
    ["A woman is cooking food.", "A man is eating something."],
    [
        "Dubai oil prices are rising.",
        "Dubai cookies are popular.",
    ],
    ["A man is biting a dog.", "A tiger is biting a cat."],
    ["John hit Minsoo.", "Minsoo hit John."],
]
MXQ_FILENAME = "stsb-bert-tiny-safetensors.mxq"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path("../../../compilation/bert/bert-mxq"),
    )
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_folder)
    model = BertMXQ(args.model_folder / MXQ_FILENAME, args.model_folder)

    try:
        print("Cosine Similarity (range: -1 to 1, higher = more similar)\n")
        for dummy_pair in DUMMY_CORPUS:
            s1 = model(**tokenizer(dummy_pair[0], return_tensors="pt"))
            s2 = model(**tokenizer(dummy_pair[1], return_tensors="pt"))
            similarity = torch.nn.functional.cosine_similarity(s1, s2, dim=1)
            print(f'  {similarity.item():.4f}  |  "{dummy_pair[0]}" vs "{dummy_pair[1]}"')
    finally:
        model.dispose()
