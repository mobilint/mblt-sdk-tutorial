import os

from qbcompiler import mxq_compile
from qbcompiler.configs import CompileConfig

mblt_path = "mblt/Qwen2-VL-2B-Instruct_text_model.mblt"
save_path = "mxq/Qwen2-VL-2B-Instruct_text_model.mxq"
calib_data_path = "../calibration/calibration_data/language/npy_files.txt"
device = "cuda"

# Ensure output directory exists
output_dir = os.path.dirname(os.path.abspath(save_path))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


compile_config = CompileConfig.model_validate(
    {
        "bit": {"layerOverrides": {"activation16Bits": ["inputs_embeds/reshape"]}},
        "equivalentTransformation": {
            "QK": {"apply": True},
            "UD": {"apply": True, "learn": True},
            "SpinR1": {"apply": True},
            "SpinR2": {"apply": True},
        },
    }
)

mxq_compile(
    mblt_path,
    save_path=save_path,
    calib_data_path=calib_data_path,
    device=device,
    inference_scheme="single",
    singlecore_compile=True,
    compile_config=compile_config,
)
