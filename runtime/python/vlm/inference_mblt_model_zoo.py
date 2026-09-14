from argparse import ArgumentParser
from pathlib import Path

from transformers import AutoModelForImageTextToText, AutoProcessor, TextStreamer, pipeline

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_FOLDER = REPO_ROOT / "compilation/vlm/prepared/aries-rb/Qwen3-VL-2B-Instruct"
DEFAULT_IMAGE = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"


if __name__ == "__main__":
    parser = ArgumentParser(description="Run Qwen3-VL inference with MXQ models")
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL_FOLDER)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default="Describe the environment and context surrounding the main subject.")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    if not args.model_folder.is_dir():
        raise FileNotFoundError(f"Model folder not found: {args.model_folder}")
    model_folder = str(args.model_folder)
    model = AutoModelForImageTextToText.from_pretrained(model_folder, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_folder, trust_remote_code=True)
    pipe = pipeline("image-text-to-text", model=model, processor=processor)
    pipe.generation_config.max_new_tokens = None

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    try:
        pipe(
            text=messages,
            generate_kwargs={
                "max_length": args.max_length,
                "streamer": TextStreamer(tokenizer=pipe.tokenizer, skip_prompt=False),
                "repetition_penalty": 1.1,
            },
        )
    finally:
        pipe.model.model.visual.dispose()
        pipe.model.model.language_model.dispose()
