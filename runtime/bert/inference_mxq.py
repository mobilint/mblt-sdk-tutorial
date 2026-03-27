from argparse import ArgumentParser

import torch
from wrapper.bert_model import BertMXQ
from transformers import BertTokenizer

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


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--mxq_path",
        type=str,
        default="../../compilation/bert/mxq/stsb-bert-tiny-safetensors.mxq",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default="../../compilation/bert/weights/weight_dict.pth",
    )
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors", trust_remote_code=True
    )

    model = BertMXQ(args.mxq_path, args.weight_path)

    print("Cosine Similarity (range: -1 to 1, higher = more similar)\n")
    for dummy_pair in DUMMY_CORPUS:
        with torch.no_grad():
            s1 = model(**tokenizer(dummy_pair[0], return_tensors="pt"))
            s2 = model(**tokenizer(dummy_pair[1], return_tensors="pt"))
            similarity = torch.nn.functional.cosine_similarity(s1, s2, dim=0)

        print(f"  {similarity.item():.4f}  |  \"{dummy_pair[0]}\" vs \"{dummy_pair[1]}\"")

    model.dispose()
