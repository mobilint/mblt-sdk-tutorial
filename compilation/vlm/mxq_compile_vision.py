import os

import torch
from qbcompiler import mxq_compile
from qbcompiler.configs import CompileConfig

mblt_path = "mblt/Qwen2-VL-2B-Instruct_vision_transformer.mblt"
save_path = "mxq/Qwen2-VL-2B-Instruct_vision_transformer.mxq"
calib_data_path = "calibration_data/vision/npy_files.txt"
device = "gpu" if torch.cuda.is_available() else "cpu"
head_out_ch_rotation_matrix_path = (
    "./spinWeight/Qwen2-VL-2B-Instruct_text_model/R1/global_rotation.pth"
)

# Ensure output directory exists
output_dir = os.path.dirname(os.path.abspath(save_path))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

compile_config = CompileConfig.model_validate(
    {
        "bit": {"layerOverrides": {"activation16Bits": ["model_merger_fc2"]}},
        "equivalentTransformation": {
            "HeadOutChRotation": {
                "apply": True,
                "matrixPath": head_out_ch_rotation_matrix_path,
            },
        },
    }
)

mxq_compile(
    mblt_path,
    save_path=save_path,
    calib_data_path=calib_data_path,
    device=device,
    inference_scheme="multi",
    compile_config=compile_config,
)
