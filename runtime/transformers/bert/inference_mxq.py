from argparse import ArgumentParser

import torch
from bertmxq import BertMXQModel
from transformers import BertTokenizer

dummy_corpus = [
    ["A man is eating food.", "A man is eating something."],
    ["A woman is cooking food.", "A man is eating something."],
    [
        "Dubai oil prices are rising.",
        "Dubai cookies are popular.",
    ],
    ["A man is biting a dog.", "A tiger is biting a cat."],
    ["John hit Minsoo.", "Minsoo hit John."],
]
tokenizer = BertTokenizer.from_pretrained(
    "sentence-transformers-testing/stsb-bert-tiny-safetensors", trust_remote_code=True
)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--mxq_path",
        type=str,
        default="../../../compilation/transformers/bert/stsb-bert-tiny-safetensors.mxq",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default="../../../compilation/transformers/bert/weight_dict.pth",
    )
    args = parser.parse_args()

    model = BertMXQModel(args.mxq_path, args.weight_path)

    for dummy_pair in dummy_corpus:
        with torch.no_grad():
            s1 = model(**tokenizer(dummy_pair[0], return_tensors="pt"))
            s2 = model(**tokenizer(dummy_pair[1], return_tensors="pt"))
            similarity = torch.nn.functional.cosine_similarity(s1, s2, dim=0)

        print(similarity)
