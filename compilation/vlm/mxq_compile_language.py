import os
from argparse import ArgumentParser

import torch
from qbcompiler import mxq_compile
from qbcompiler.configs import BitConfig, EquivalentTransformationConfig


def get_device_inference_scheme(target_device):
    # REGULUS only supports the single scheme; ARIES supports all schemes in one model.
    if "regulus" in target_device:
        return "single"
    elif "aries" in target_device:
        return "all"
    raise ValueError(f"{target_device} not supported in current qbcompiler version")


if __name__ == "__main__":
    parser = ArgumentParser(description="Compile Qwen2-VL language model MBLT to MXQ")
    parser.add_argument(
        "--target-device",
        type=str,
        choices=["regulus-ra", "regulus-rb", "aries-rb"],
        default="aries-rb",
        help="Target NPU (e.g. aries-rb, regulus-rb)",
    )
    args = parser.parse_args()

    mblt_path = "mblt/Qwen2-VL-2B-Instruct_text_model.mblt"
    save_path = "mxq/Qwen2-VL-2B-Instruct_text_model.mxq"
    calib_data_path = "calibration_data/language/npy_files.txt"
    device = "gpu" if torch.cuda.is_available() else "cpu"

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(save_path))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bit_config = BitConfig.model_validate({"layerOverrides": {"activation16Bits": ["inputs_embeds/reshape"]}})
    equivalent_transformation_config = EquivalentTransformationConfig.model_validate(
        {
            "QK": {"apply": True},
            "UD": {"apply": True, "learn": True},
            "SpinR1": {"apply": True},
            "SpinR2": {"apply": True},
        }
    )

    mxq_compile(
        mblt_path,
        target_device=args.target_device,
        save_path=save_path,
        calib_data_path=calib_data_path,
        device=device,
        inference_scheme=get_device_inference_scheme(args.target_device),
        bit_config=bit_config,
        equivalent_transformation_config=equivalent_transformation_config,
    )
