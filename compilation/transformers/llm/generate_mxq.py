import os
from argparse import ArgumentParser

from qbcompiler import (
    BitConfig,
    CalibrationConfig,
    LlmConfig,
    get_advanced_quantization_config,
    get_equivalent_transformation_config,
    get_quantization_config,
    get_search_weight_scale_config,
    mxq_compile,
)
from transformers import AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

EXP_PRESETS = {
    "w8": {
        "bit": {
            "query_weight_bits": 8,
            "key_weight_bits": 8,
            "value_weight_bits": 8,
            "output_weight_bits": 8,
            "ffn_weight_bits": 8,
            "head_weight_bits": 8,
        },
        "sws": {
            "apply": False,
        },
        "et": {},
    },
    "w4": {
        "bit": {
            "query_weight_bits": 4,
            "key_weight_bits": 4,
            "value_weight_bits": 4,
            "output_weight_bits": 4,
            "ffn_weight_bits": 4,
            "head_weight_bits": 4,
        },
        "sws": {
            "apply": True,
            "query": True,
            "key": True,
            "value": True,
            "out": True,
            "ffn": True,
        },
        "et": {
            "norm_conv_apply": True,
            "qk_apply": True,
            "ud_apply": True,
            "vo_apply": True,
            "ff_multi_lut_apply": True,
            "spin_r1_apply": True,
            "spin_r2_apply": True,
            "qk_rotation_apply": True,
            "optimize_ffn_apply": True,
        },
    },
    "w4v8": {
        "bit": {
            "query_weight_bits": 4,
            "key_weight_bits": 4,
            "value_weight_bits": 8,
            "output_weight_bits": 4,
            "ffn_weight_bits": 4,
            "head_weight_bits": 4,
        },
        "sws": {
            "apply": True,
            "query": True,
            "key": True,
            "value": True,
            "out": True,
            "ffn": True,
        },
        "et": {
            "norm_conv_apply": True,
            "qk_apply": True,
            "ud_apply": True,
            "vo_apply": True,
            "ff_multi_lut_apply": True,
            "spin_r1_apply": True,
            "spin_r2_apply": True,
            "qk_rotation_apply": True,
            "optimize_ffn_apply": True,
        },
    },
}
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default="meta-llama/Llama-3.2-1B-Instruct"
    )
    parser.add_argument(
        "--calib-data-path",
        type=str,
        default="./calib/datas/meta-llama-Llama-3.2-1B-Instruct/en",
    )
    parser.add_argument("--save-path", type=str, default="./Llama-3.2-1B-Instruct.mxq")
    parser.add_argument("--bit", type=str, default="w8", choices=["w8", "w4v8", "w4"])
    args = parser.parse_args()

    calib_config = CalibrationConfig.from_kwargs(
        quantization_method=1,  # 0 for per tensor, 1 for per channel
        quantization_output=0,  # 0 for layer, 1 for channel
        quantization_mode=2,  # maxpercentile
        percentile=0.9999,
        topk_ratio=0.01,
    )
    bit_config = BitConfig.from_kwargs(**EXP_PRESETS[args.bit]["bit"])
    quant_config = get_quantization_config(
        calibration=calib_config,
        bit=bit_config,
    )

    llm_config = LlmConfig.from_kwargs(
        llm_config_apply=True,
        max_data_length=4096,
        max_sequence_length=4096,
        max_cache_length=4096,
        max_core_data_length=128,
        use_full_seq_length=True,
        batch_size=1,  # mxq for one batch
        npu_core_ids=[0],  # assigne cluster0, core0
    )
    sws_config = get_search_weight_scale_config(**EXP_PRESETS[args.bit]["sws"])
    et_config = get_equivalent_transformation_config(**EXP_PRESETS[args.bit]["et"])

    adv_quant_config = get_advanced_quantization_config(
        equivalent_transformation=et_config,
        search_weight_scale=sws_config,
    )

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

    mxq_compile(
        model=args.model_path,
        calib_data_path=args.calib_data_path,
        save_path=args.save_path,
        backend="torch",
        device="gpu",  # using GPU is recommended
        llm_config=llm_config,
        quantization_config=quant_config,
        advanced_quantization_config=adv_quant_config,
        hf_config=hf_config,
    )

    print("Model compiled successfully.")
