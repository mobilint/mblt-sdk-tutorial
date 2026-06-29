import os
from argparse import ArgumentParser

from qbcompiler import CalibrationConfig, mxq_compile
from qbcompiler.model_dict.parser.backend.torch.object_wrapper import set_attention_mask
from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor
from transformers import BertModel, BertTokenizer

def get_device_inference_scheme(target_device):
    # REGULUS only supports the single scheme; ARIES supports all schemes in one model.
    if "regulus" in target_device:
        return "single"
    elif "aries" in target_device:
        return "all"
    raise ValueError(f"{target_device} not supported in current qbcompiler version")


if __name__ == "__main__":
    parser = ArgumentParser(description="Compile Sentence-BERT to MXQ with quantization")
    parser.add_argument(
        "--target-device",
        type=str,
        default="aries-rb",
        help="Target NPU (e.g. aries-rb, regulus-rb)",
    )
    parser.add_argument(
        "--calib-data-path",
        type=str,
        default="./calibration_data",
        help="Path to the calibration data",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./mxq/stsb-bert-tiny-safetensors.mxq",
        help="Path to save the MXQ model",
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

    calibration_config = CalibrationConfig(
        method=1,  # WChAMulti: weight per-channel, activation multi-layer
        output=0,  # Layer: per-layer output quantization
        mode=1,  # MaxPercentile
        max_percentile=CalibrationConfig.MaxPercentile(
            percentile=0.999,
            topk_ratio=0.01,
        ),
    )

    # inference scheme differs device by device
    inference_scheme = get_device_inference_scheme(args.target_device)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    mxq_compile(
        model=model,
        target_device=args.target_device,
        save_path=args.save_path,
        calib_data_path=args.calib_data_path,
        backend="torch",
        feed_dict=feed_dict,
        inference_scheme=inference_scheme,
        calibration_config=calibration_config,
    )
