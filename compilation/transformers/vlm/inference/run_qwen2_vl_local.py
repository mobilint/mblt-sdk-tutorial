from transformers import TextStreamer
from mblt_model_zoo.transformers import pipeline, AutoProcessor
from PIL import Image

# Load processor
processor = AutoProcessor.from_pretrained(".", use_fast=True)

# Create pipeline
pipe = pipeline(
    "image-text-to-text",
    model=".",
    processor=processor,
)

# Remove max_new_tokens limit
pipe.generation_config.max_new_tokens = None

# Prepare messages with image
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe the environment and context surrounding the main subject."},
        ],
    }
]

# Run inference
pipe(
    text=messages,
    generate_kwargs={
        "max_length": 512,
        "streamer": TextStreamer(tokenizer=pipe.tokenizer, skip_prompt=False),
    },
)

# Clean up
pipe.model.dispose()
