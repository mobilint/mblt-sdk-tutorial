from argparse import ArgumentParser

import torch
from transformers import AutoConfig, AutoTokenizer, TextStreamer
from wrapper.llama_model import LlamaMXQ

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


if __name__ == "__main__":
    parser = ArgumentParser(description="LLM Inference using local wrapper")
    parser.add_argument(
        "--mxq-path",
        type=str,
        default="../../../compilation/llm/Llama-3.2-1B-Instruct-W8.mxq",
        help="Path to the compiled MXQ file",
    )
    parser.add_argument(
        "--embedding-path",
        type=str,
        default="../../../compilation/llm/llama-mxq-w8/model.safetensors",
        help="Path to the embedding weight file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain the concept of NPU, compared to GPU and CPU, in 3 short bullet points.",
        help="User prompt for the model",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )
    args = parser.parse_args()

    device = "cpu"

    config = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, config=config)
    model = LlamaMXQ(
        config=config,
        mxq_path=args.mxq_path,
        embedding_path=args.embedding_path,
        max_sub_seq=192,
    )

    model.to(device)
    model.eval()

    chat = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": args.prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer([prompt_text], return_tensors="pt").to(device)

    streamer = TextStreamer(tokenizer, skip_prompt=True)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("=== MODEL OUTPUT ===")
    print(generated_text)
