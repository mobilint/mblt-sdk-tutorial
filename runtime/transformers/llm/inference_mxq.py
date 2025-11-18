import torch
from transformers import AutoTokenizer
from transformers import AutoConfig
from llamamxq import LlamaMXQ


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
config = AutoConfig.from_pretrained(MODEL_NAME)

def main():
    device = "cpu" # Do not use gpu since we are using npu.

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, config=config)
    model = LlamaMXQ(config=config, mxq_path="/workspace/tutorial/Llama-3.2-1B-Instruct.mxq", embedding_weight_path="/workspace/tutorial/embedding.pt", max_sub_seq=192)

    model.to(device)
    model.eval()

    user_prompt = "Explain the concept of NPU, compared to GPU and CPU, in 3 short bullet points."

    # (Optional) If your tokenizer has a chat template, you can use it:
    chat = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("=== MODEL OUTPUT ===")
    print(generated_text)

if __name__ == "__main__":
    main()