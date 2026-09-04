import argparse

import mblt_model_zoo.hf_transformers.models.llama.modeling_llama  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def main():
    parser = argparse.ArgumentParser(description="LLM Inference using mblt-model-zoo")
    parser.add_argument(
        "--model-folder",
        type=str,
        default="../../../compilation/llm/llama-mxq-w8",
        help="Path to the prepared model folder",
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

    print(f"Loading model from {args.model_folder}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_folder)
    tokenizer = AutoTokenizer.from_pretrained(args.model_folder)

    chat = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": args.prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors="pt")

    streamer = TextStreamer(tokenizer, skip_prompt=True)

    print("Running inference...")
    model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id,
    )

    model.dispose()


if __name__ == "__main__":
    main()
