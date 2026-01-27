from qbcompiler import (
    get_advanced_quantization_config,
    get_bit_config,
    get_calibration_config,
    get_equivalent_transformation_config,
    get_quantization_config,
    mxq_compile,
)

mblt_path = "mblt/Qwen2-VL-2B-Instruct_text_model.mblt"
save_path = "mxq/Qwen2-VL-2B-Instruct_text_model.mxq"
calib_data_path = "../calibration/calibration_data/language/npy_files.txt"
device = "cuda"

cal_config = get_calibration_config()
bit_config = get_bit_config(activation_16bits=["inputs_embeds/reshape"])
quantization_config = get_quantization_config(cal_config, bit_config)

et_config = get_equivalent_transformation_config(
    qk_apply=True,
    ud_apply=True,
    ud_learn=True,
    spin_r1_apply=True,
    spin_r2_apply=True,
)

advanced_config = get_advanced_quantization_config(equivalent_transformation=et_config)

mxq_compile(
    mblt_path,
    save_path=save_path,
    calib_data_path=calib_data_path,
    device=device,
    inference_scheme="single",
    singlecore_compile=True,
    quantization_config=quantization_config,
    advanced_quantization_config=advanced_config,
)
