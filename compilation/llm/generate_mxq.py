"""Compile LLM to MXQ format with 8-bit quantization."""

from argparse import ArgumentParser

import torch
from qbcompiler import (
    BitConfig,
    CalibrationConfig,
    LlmConfig,
    mxq_compile,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument(
        "--calib-data-path",
        type=str,
        default="./calibration_data/datas/meta-llama-Llama-3.2-1B-Instruct/en",
    )
    parser.add_argument("--save-path", type=str, default="./Llama-3.2-1B-Instruct.mxq")
    args = parser.parse_args()

    device = "gpu" if torch.cuda.is_available() else "cpu"

    calib_config = CalibrationConfig(
        method=1,  # WChAMulti: weight per-channel, activation multi-layer
        output=0,  # per-layer output quantization
        mode=1,  # MaxPercentile
    )

    # All transformer components quantized to 8-bit
    bit_config = BitConfig(
        transformer=BitConfig.Transformer(
            weight=BitConfig.Transformer.Weight(
                query=8,
                key=8,
                value=8,
                output=8,
                ffn=8,
                head=8,
            ),
        )
    )

    llm_config = LlmConfig(
        apply=True,
        attributes=LlmConfig.Attributes(
            max_data_length=4096,
            max_sequence_length=4096,
            max_cache_length=4096,
            max_core_data_length=128,
            calibration=LlmConfig.Attributes.Calibration(use_full_seq_length=True),
            runtime=LlmConfig.Attributes.Runtime(batch_size=1, npu_core_ids=[0]),
        ),
    )

    mxq_compile(
        model=args.model_path,
        calib_data_path=args.calib_data_path,
        save_path=args.save_path,
        backend="torch",
        device=device,
        calibration_config=calib_config,
        bit_config=bit_config,
        llm_config=llm_config,
        hf_config={
            "library": "transformers",
            "loader": "AutoModelForCausalLM",
            "tokenizer": "AutoTokenizer",
            "model_args": (),
            "model_kwargs": {"trust_remote_code": True},
            "tokenizer_args": (),
            "tokenizer_kwargs": {"trust_remote_code": True},
        },
    )

    print("Model compiled successfully.")
