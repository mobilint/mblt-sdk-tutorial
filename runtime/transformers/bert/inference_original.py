import torch
from transformers import BertModel, BertTokenizer

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
    "sentence-transformers/msmarco-bert-base-dot-v5", trust_remote_code=True
)
model = BertModel.from_pretrained(
    "sentence-transformers/msmarco-bert-base-dot-v5", trust_remote_code=True
)
model.eval()


if __name__ == "__main__":
    # dummy corpus STS

    for dummy_pair in dummy_corpus:
        with torch.no_grad():
            s1 = model(tokenizer(dummy_pair[0], return_tensors="pt")["input_ids"])
            s2 = model(tokenizer(dummy_pair[1], return_tensors="pt")["input_ids"])
            similarity = torch.nn.functional.cosine_similarity(
                s1["pooler_output"], s2["pooler_output"], dim=1
            )
        print(similarity)
