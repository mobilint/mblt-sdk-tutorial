"""Compile the Qwen3-VL language model MBLT to MXQ (decoder, benchmark-best 2B config).

This run applies 8-bit weights, 16-bit activations on inputs_embeds/deepstack,
SpinR1/SpinR2 rotations, and a weight-scale search. It GENERATES the SpinR1
rotation matrix at:
    ./spinWeight/Qwen3-VL-2B-Instruct_text_model/R1/global_rotation.pth
which the vision (encoder) MXQ compilation and get_safetensors.py reuse. Run this
BEFORE mxq_compile_vision.py.
"""

import os
from argparse import ArgumentParser

import torch
from qbcompiler import mxq_compile
from qbcompiler.configs import (
    BitConfig,
    CalibrationConfig,
    EquivalentTransformationConfig,
    LlmConfig,
    OptqConfig,
    ResourceManagementConfig,
    SearchWeightScaleConfig,
)

TARGET_DEVICES = ["regulus-ra", "aries-rb", "regulus-rb"]


def get_device_inference_scheme(target_device):
    # REGULUS only supports the single scheme; ARIES supports all schemes in one model.
    if "regulus" in target_device:
        return "single"
    elif "aries" in target_device:
        return "all"
    raise ValueError(f"{target_device} not supported in current qbcompiler version")


if __name__ == "__main__":
    parser = ArgumentParser(description="Compile Qwen3-VL language model MBLT to MXQ")
    parser.add_argument(
        "--target-device",
        type=str,
        required=True,
        choices=TARGET_DEVICES,
        help="Target NPU (e.g. aries-rb, regulus-rb)",
    )
    args = parser.parse_args()

    mblt_path = "mblt/Qwen3-VL-2B-Instruct_text_model.mblt"
    save_path = "mxq/Qwen3-VL-2B-Instruct_text_model.mxq"
    calib_data_path = "calibration_data/language/npy_files.json"
    device = "gpu" if torch.cuda.is_available() else "cpu"

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(save_path))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    calibration_config = CalibrationConfig(output=0, mode=0)

    bit_config = BitConfig(
        layer_overrides=BitConfig.LayerOverrides(
            activation_16bits=[
                "inputs_embeds/reshape",
                "deepstack_visual_embeds_0",
            ],
        ),
    )

    resource_management_config = ResourceManagementConfig(
        weight_dtype="float32",
        use_gpu_only_for_calibration=True,
        weight_memory=ResourceManagementConfig.WeightMemory(method=1),
    )

    llm_config = LlmConfig(
        apply=True,
        attributes=LlmConfig.Attributes(
            calibration=LlmConfig.Attributes.Calibration(use_full_seq_length=True),
        ),
    )

    equivalent_transformation_config = EquivalentTransformationConfig(
        qk=EquivalentTransformationConfig.Qk(apply=False),
        ud=EquivalentTransformationConfig.Ud(apply=True, smoothing_factor=0.8),
        vo=EquivalentTransformationConfig.Vo(apply=True),
        spin_r1=EquivalentTransformationConfig.SpinR1(apply=True),
        spin_r2=EquivalentTransformationConfig.SpinR2(apply=True),
        optimize_ffn=EquivalentTransformationConfig.OptimizeFfn(apply=True),
    )

    optq_config: OptqConfig | None = None

    search_weight_scale_config = SearchWeightScaleConfig(
        apply=True,
        transformer=SearchWeightScaleConfig.Transformer(
            query=True,
            key=True,
            value=True,
            out=True,
            ffn=True,
        ),
    )

    mxq_compile(
        mblt_path,
        target_device=args.target_device,
        save_path=save_path,
        calib_data_path=calib_data_path,
        device=device,
        inference_scheme=get_device_inference_scheme(args.target_device),
        calibration_config=calibration_config,
        bit_config=bit_config,
        resource_management_config=resource_management_config,
        llm_config=llm_config,
        equivalent_transformation_config=equivalent_transformation_config,
        optq_config=optq_config,
        search_weight_scale_config=search_weight_scale_config,
    )
