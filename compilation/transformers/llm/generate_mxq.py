import argparse
import os

from qbcompiler import mxq_compile

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-path", type=str, default="meta-llama/Llama-3.2-1B-Instruct"
)
parser.add_argument(
    "--calib-data-path",
    type=str,
    default="./calib/datas/meta-llama-Llama-3.2-1B-Instruct/en",
)
parser.add_argument("--save-path", type=str, default="./Llama-3.2-1B-Instruct.mxq")
args = parser.parse_args()

hf_config = {
    "library": "transformers",
    "loader": "AutoModelForCausalLM",
    "tokenizer": "AutoTokenizer",
    "model_args": (),
    "model_kwargs": {
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True,
    },
    "tokenizer_args": (),
    "tokenizer_kwargs": {},
}

mxq_compile(
    model=args.model_path,
    calib_data_path=args.calib_data_path,
    save_path=args.save_path,
    backend="torch",
    singlecore_compile=True,
    quantization_output=0,  # Layer
    hf_config=hf_config,
    use_gpu_only_for_calibration=True,
    weight_dtype="bfloat16",
    device="gpu",  # using GPU is recommended
)
