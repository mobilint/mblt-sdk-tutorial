import argparse

import mblt_model_zoo.hf_transformers.models.llama.modeling_llama  # noqa: F401
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def main():
    parser = argparse.ArgumentParser(description="LLM Inference using mblt-model-zoo")
    parser.add_argument(
        "--model-folder",
        type=str,
        default="./llama-mxq",
        help="Path to the prepared model folder",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="mobilint/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID for tokenizer download",
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

    # Load model from config.json in model folder.
    # config.json contains MXQ path and NPU core allocation (target_cores).
    # Core mode can be changed by editing config.json — see README for details.
    print(f"Loading model from {args.model_folder}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_folder)

    # Load tokenizer from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Prepare chat messages
    chat = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": args.prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors="pt")

    streamer = TextStreamer(tokenizer, skip_prompt=True)

    # Generate
    print("Running inference...")
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Clean up NPU resources
    model.dispose()


if __name__ == "__main__":
    main()
