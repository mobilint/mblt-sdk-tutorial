from argparse import ArgumentParser
from qubee import mxq_compile, get_llm_config
import torch

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./Llama-3.2-1B-Instruct")
    parser.add_argument(
        "--calib_data_path", type=str, default="./Llama-3.2-1B-Instruct-Wikipedia-en"
    )
    parser.add_argument("--save_path", type=str, default="./Llama-3.2-1B-Instruct.mxq")
    args = parser.parse_args()

    model_path = args.model_path
    calib_data_path = args.calib_data_path
    save_path = args.save_path

    hf_config = {
        "library": "transformers",
        "loader": "AutoModelForCausalLM",
        "tokenizer": "AutoTokenizer",
        "model_args": (),
        "model_kwargs": {
            "trust_remote_code": True,
        },
        "tokenizer_args": (),
        "tokenizer_kwargs": {
            "trust_remote_code": True,
        },
    }

    # seqlen should be the same as cachelen in the current version
    llm_config = get_llm_config(
        max_seq_len=4096,
        max_cache_len=4096,
        use_full_seq_len_calib=True,
    )

    mxq_compile(
        model=args.model_path,
        calib_data_path=args.calib_data_path,
        save_path=args.save_path,
        backend="torch",
        singlecore_compile=True,  # required for large models with cache support
        quant_output="layer",
        hf_config=hf_config,
        use_gpu_only_for_calibration=torch.cuda.is_available(),  # decreases GPU memory usage
        weight_dtype="bfloat16",  # calibration weight data type that should be the same as the dtype of the original model
        llm_config=llm_config,
        device=(
            "gpu" if torch.cuda.is_available() else "cpu"
        ),  # using a GPU is recommended
    )
