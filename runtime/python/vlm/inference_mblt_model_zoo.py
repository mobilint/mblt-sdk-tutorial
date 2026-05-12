import argparse

import mblt_model_zoo.hf_transformers.models.qwen2_vl.modeling_qwen2_vl  # noqa: F401
from transformers import AutoModelForImageTextToText, AutoProcessor, TextStreamer, pipeline


def main():
    parser = argparse.ArgumentParser(description="VLM Inference using mblt-model-zoo")
    parser.add_argument(
        "--model-folder",
        type=str,
        default="./qwen2-vl-mxq",
        help="Path to the prepared model folder",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="mobilint/Qwen2-VL-2B-Instruct",
        help="HuggingFace model ID for processor download",
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

    # Load model from config.json in model folder.
    # config.json contains MXQ paths, NPU core allocation (target_cores),
    # and _name_or_path for automatic processor download.
    # Core mode can be changed by editing config.json — see README for details.
    print(f"Loading model from {args.model_folder}...")
    model = AutoModelForImageTextToText.from_pretrained(args.model_folder)

    # Load processor from HuggingFace.
    # trust_remote_code=True is required because mobilint/Qwen2-VL-2B-Instruct
    # uses a custom processor (MobilintQwen2VLProcessor) hosted on HuggingFace.
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

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
