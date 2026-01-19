from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("jhgan/ko-sbert-sts", trust_remote_code=True)

print(
    tokenizer("아버지가 광주 행정센터에 방문하셨다.", return_tensors="pt")["input_ids"]
)
