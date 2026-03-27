import os

from qbcompiler import CalibrationConfig, mxq_compile
from qbcompiler.model_dict.parser.backend.torch.object_wrapper import set_attention_mask
from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor
from transformers import BertModel, BertTokenizer

CALIB_PATH = "./calibration_data"
MXQ_PATH = "./mxq/stsb-bert-tiny-safetensors.mxq"

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

    calibration_config = CalibrationConfig(
        method=1,  # WChAMulti: weight per-channel, activation multi-layer
        output=0,  # Layer: per-layer output quantization
        mode=1,  # MaxPercentile
        max_percentile=CalibrationConfig.MaxPercentile(
            percentile=0.999,
            topk_ratio=0.01,
        ),
    )

    os.makedirs(os.path.dirname(MXQ_PATH), exist_ok=True)
    mxq_compile(
        model=model,
        save_path=MXQ_PATH,
        calib_data_path=CALIB_PATH,
        backend="torch",
        feed_dict=feed_dict,
        calibration_config=calibration_config,
    )
