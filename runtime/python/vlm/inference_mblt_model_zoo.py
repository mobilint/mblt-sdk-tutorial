import argparse

from transformers import AutoModelForImageTextToText, AutoProcessor, TextStreamer, pipeline


def main():
    parser = argparse.ArgumentParser(description="VLM Inference using a self-contained MXQ model folder")
    parser.add_argument(
        "--model-folder",
        type=str,
        default="./Qwen3-VL-2B-Instruct",
        help="Path to the self-contained model folder (config, proxy, tokenizer, MXQ).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        help="Path or URL to the input image",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the environment and context surrounding the main subject.",
        help="Text prompt for the model",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum generation length",
    )

    args = parser.parse_args()

    # Load the model and processor from the same local folder.
    # The folder is self-contained: config.json's auto_map points at the bundled
    # proxy classes, so trust_remote_code=True loads everything from here (no
    # separate HuggingFace model-id needed). config.json holds the MXQ paths and
    # NPU core allocation.
    print(f"Loading model from {args.model_folder}...")
    model = AutoModelForImageTextToText.from_pretrained(args.model_folder, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_folder, trust_remote_code=True)

    # Create pipeline
    pipe = pipeline(
        "image-text-to-text",
        model=model,
        processor=processor,
    )
    # Disable max_new_tokens so that max_length (passed via generate_kwargs) controls
    # the generation length. Without this, the default max_new_tokens would override max_length.
    pipe.generation_config.max_new_tokens = None

    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    # Run inference with streaming output
    print("Running inference...")
    pipe(
        text=messages,
        generate_kwargs={
            "max_length": args.max_length,
            "streamer": TextStreamer(tokenizer=pipe.tokenizer, skip_prompt=False),
            "repetition_penalty": 1.1,
        },
    )

    # Clean up NPU resources
    pipe.model.model.visual.dispose()
    pipe.model.model.language_model.dispose()


if __name__ == "__main__":
    main()
