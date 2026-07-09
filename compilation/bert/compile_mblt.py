import os
from argparse import ArgumentParser

from qbcompiler import mblt_compile
from qbcompiler.model_dict.parser.backend.torch.object_wrapper import set_attention_mask
from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor
from transformers import BertModel, BertTokenizer

if __name__ == "__main__":
    parser = ArgumentParser(description="Compile Sentence-BERT to MBLT intermediate format")
    parser.add_argument(
        "--target-device",
        type=str,
        choices=["regulus-rb", "aries-rb"],
        default="aries-rb",
        help="Target NPU (e.g. aries-rb, regulus-rb)",
    )
    parser.add_argument(
        "--mblt-path",
        type=str,
        default="./mblt/stsb-bert-tiny-safetensors.mblt",
        help="Path to save the MBLT model",
    )
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        trust_remote_code=True,
    )
    model = BertModel.from_pretrained(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        trust_remote_code=True,
    )
    model.eval()

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    feed_dict = {}
    for k, v in inputs.items():
        wrapped = wrap_tensor(k, v)
        wrapped.src_shape[1].set_dynamic()
        feed_dict[k] = wrapped
    set_attention_mask(feed_dict["attention_mask"], "padding_mask")

    os.makedirs(os.path.dirname(args.mblt_path), exist_ok=True)
    mblt_compile(
        model=model,
        mblt_save_path=args.mblt_path,
        target_device=args.target_device,
        backend="torch",
        feed_dict=feed_dict,
        cpu_offload=True,
    )
