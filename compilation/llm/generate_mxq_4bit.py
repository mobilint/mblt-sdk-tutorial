"""Compile LLM to MXQ format with 4-bit quantization.

Uses SpinQuant rotation and weight scale search to compensate for
aggressive 4-bit quantization loss. Generates spinWeight/ rotation
matrices required for embedding rotation at inference.
"""

from argparse import ArgumentParser

import torch
from qbcompiler import (
    BitConfig,
    CalibrationConfig,
    EquivalentTransformationConfig,
    LlmConfig,
    SearchWeightScaleConfig,
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
    parser.add_argument("--bit", type=str, default="w4", choices=["w4", "w4v8"])
    args = parser.parse_args()

    device = "gpu" if torch.cuda.is_available() else "cpu"

    calib_config = CalibrationConfig(
        method=1,  # WChAMulti: weight per-channel, activation multi-layer
        output=0,  # per-layer output quantization
        mode=1,  # MaxPercentile
    )

    # w4: all 4-bit, w4v8: 4-bit except value (kept at 8-bit for accuracy)
    value_bit = 8 if args.bit == "w4v8" else 4
    bit_config = BitConfig(
        transformer=BitConfig.Transformer(
            weight=BitConfig.Transformer.Weight(
                query=4,
                key=4,
                value=value_bit,
                output=4,
                ffn=4,
                head=4,
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

    # Learn per-layer weight scales to compensate for 4-bit quantization error
    sws_config = SearchWeightScaleConfig(
        apply=True,
        transformer=SearchWeightScaleConfig.Transformer(
            query=True,
            key=True,
            value=True,
            out=True,
            ffn=True,
        ),
    )

    # SpinQuant rotation + equivalent transformations to reduce quantization loss.
    # Generates spinWeight/ rotation matrices for embedding rotation at inference.
    et_config = EquivalentTransformationConfig(
        norm_conv=EquivalentTransformationConfig.NormConv(apply=True),
        qk=EquivalentTransformationConfig.Qk(apply=True),
        ud=EquivalentTransformationConfig.Ud(apply=True),
        vo=EquivalentTransformationConfig.Vo(apply=True),
        feed_forward_multi_lut=EquivalentTransformationConfig.FeedForwardMultiLut(apply=True),
        spin_r1=EquivalentTransformationConfig.SpinR1(apply=True),
        spin_r2=EquivalentTransformationConfig.SpinR2(apply=True),
        qk_rotation=EquivalentTransformationConfig.QkRotation(apply=True),
        optimize_ffn=EquivalentTransformationConfig.OptimizeFfn(apply=True),
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
        search_weight_scale_config=sws_config,
        equivalent_transformation_config=et_config,
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
