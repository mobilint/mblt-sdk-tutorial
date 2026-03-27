import os

from qbcompiler import mblt_compile
from qbcompiler.model_dict.parser.backend.torch.object_wrapper import set_attention_mask
from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor
from transformers import BertModel, BertTokenizer

MBLT_PATH = "./mblt/stsb-bert-tiny-safetensors.mblt"

if __name__ == "__main__":
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

    os.makedirs(os.path.dirname(MBLT_PATH), exist_ok=True)
    mblt_compile(
        model=model,
        mblt_save_path=MBLT_PATH,
        backend="torch",
        feed_dict=feed_dict,
        cpu_offload=True,
    )
