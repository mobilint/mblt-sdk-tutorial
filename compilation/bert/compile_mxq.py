from qbcompiler import QuantizationConfig, mxq_compile
from qbcompiler.model_dict.parser.backend.torch.object_wrapper import set_attention_mask
from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor
from transformers import BertModel, BertTokenizer

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

    quantization_config = QuantizationConfig.from_kwargs(
        quantization_method=1,  # 0 for per tensor, 1 for per channel
        quantization_output=0,  # 0 for layer, 1 for channel
        quantization_mode=1,  # maxpercentile
        percentile=0.999,  # quantization percentile
        topk_ratio=0.01,  # quantization topk
    )

    mxq_compile(
        model=model,
        save_path="stsb-bert-tiny-safetensors.mxq",
        calib_data_path="./calib",
        backend="torch",
        feed_dict=feed_dict,
        quantization_config=quantization_config,
    )
