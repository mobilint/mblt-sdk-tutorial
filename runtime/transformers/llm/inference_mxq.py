from argparse import ArgumentParser

import torch
from llamamxq import LlamaMXQ
from transformers import AutoConfig, AutoTokenizer, TextStreamer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
config = AutoConfig.from_pretrained(MODEL_NAME)


def main(mxq_path, embedding_weight_path):

    device = "cpu"  # Do not use gpu since we are using npu.

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, config=config)
    model = LlamaMXQ(
        config=config,
        mxq_path=mxq_path,
        embedding_weight_path=embedding_weight_path,
        max_sub_seq=192,
    )

    model.to(device)
    model.eval()

    user_prompt = (
        "Explain the concept of NPU, compared to GPU and CPU, in 3 short bullet points."
    )

    # (Optional) If your tokenizer has a chat template, you can use it:
    chat = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer([prompt_text], return_tensors="pt").to(device)

    streamer = TextStreamer(tokenizer, skip_prompt=True)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("=== MODEL OUTPUT ===")
    print(generated_text)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--mxq-path",
        type=str,
        default="../../../compilation/transformers/llm/Llama-3.2-1B-Instruct_w8.mxq",
    )
    parser.add_argument(
        "--embedding-weight-path",
        type=str,
        default="../../../compilation/transformers/llm/embedding.pt",
    )
    args = parser.parse_args()
    main(args.mxq_path, args.embedding_weight_path)
