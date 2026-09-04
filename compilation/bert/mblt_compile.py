from argparse import ArgumentParser
from pathlib import Path

import torch
from qbcompiler import mblt_compile
from transformers import BertModel, BertTokenizer


class BertBody(torch.nn.Module):
    def __init__(self, model: BertModel):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.encoder(hidden_states, return_dict=False)[0]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="sentence-transformers-testing/stsb-bert-tiny-safetensors",
    )
    parser.add_argument(
        "--target-device",
        choices=["regulus-rb", "aries-rb"],
        default="aries-rb",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("./mblt/stsb-bert-tiny-safetensors.mblt"),
    )
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_id)
    source_model = BertModel.from_pretrained(args.model_id, attn_implementation="eager").eval()
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    with torch.inference_mode():
        hidden_states = source_model.embeddings(
            input_ids=inputs["input_ids"],
            token_type_ids=inputs["token_type_ids"],
        )
    model = BertBody(source_model).eval()

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    mblt_compile(
        model=model,
        mblt_save_path=str(args.save_path),
        target_device=args.target_device,
        backend="torch",
        feed_dict={"hidden_states": hidden_states},
        dynamic_axes={"hidden_states": [1]},
    )
