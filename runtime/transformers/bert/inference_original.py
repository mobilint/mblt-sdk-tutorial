import torch
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

dummy_corpus = [
    ["한 남자가 음식을 먹고 있다.", "한 남자가 무언가를 먹고 있다."],
    ["한 여성이 고기를 요리하고 있다.", "한 남자가 말하고 있다."],
    [
        "두바이산 원유의 배럴당 가격이 폭증하고 있다.",
        "두바이 쫀득 쿠키가 유행하고 있다.",
    ],
    ["어떤 남자가 개한테 물려서 다쳤다.", "어떤 호랑이가 고양이에게 물려서 다쳤다."],
    ["영수가 민수를 때렸다.", "민수가 영수한테 맞았다."],
]
tokenizer = BertTokenizer.from_pretrained("jhgan/ko-sbert-sts", trust_remote_code=True)
model = BertModel.from_pretrained("jhgan/ko-sbert-sts", trust_remote_code=True)
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
