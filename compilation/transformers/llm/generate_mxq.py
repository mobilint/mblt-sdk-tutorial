import argparse
import os

from qubee import mxq_compile

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path", type=str, default="meta-llama/Llama-3.2-1B-Instruct"
)
parser.add_argument(
    "--calib_data_path",
    type=str,
    default="/workspace/tutorial/calib/datas/meta-llama-Llama-3.2-1B-Instruct/en",
)
parser.add_argument(
    "--save_path", type=str, default="/workspace/tutorial/Llama-3.2-1B-Instruct.mxq"
)
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
    model=model_path,
    calib_data_path=calib_data_path,
    save_path=save_path,
    backend="torch",
    singlecore_compile=True,
    quant_output="layer",
    hf_config=hf_config,
    use_gpu_only_for_calibration=True,
    weight_dtype="bfloat16",
    device="gpu",  # using GPU is recommended
)
