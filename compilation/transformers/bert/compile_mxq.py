from qubee import mxq_compile
from qubee.model_dict.parser.backend.torch.object_wrapper import set_attention_mask
from qubee.model_dict.parser.backend.torch.util import wrap_tensor
from transformers import BertModel, BertTokenizer

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(
        "jhgan/ko-sbert-sts",
        trust_remote_code=True,
    )
    model = BertModel.from_pretrained("jhgan/ko-sbert-sts", trust_remote_code=True)
    model.eval()

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    feed_dict = {}
    for k, v in inputs.items():
        wrapped = wrap_tensor(k, v)
        wrapped.src_shape[1].set_dynamic()
        feed_dict[k] = wrapped
    set_attention_mask(feed_dict["attention_mask"], "padding_mask")

    mxq_compile(
        model=model,
        save_path="ko-sbert-sts.mxq",
        calib_data_path="./calib",
        backend="torch",
        feed_dict=feed_dict,
    )
