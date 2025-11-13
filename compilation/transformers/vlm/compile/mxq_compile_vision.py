from qubee import mxq_compile
from qubee import get_advanced_quantization_config, get_equivalent_transformation_config, get_bit_config, get_calibration_config, get_quantization_config, get_llm_config

mblt_path = 'qwen2vl_vision.mblt'
save_path = 'qwen2vl_vision.mxq'
calib_data_path = '/workspace/data_prep/calibration_data/vision/npy_files.txt'
device='cuda'
head_out_ch_rotation_matrix_path = "/tmp/qubee/spinWeight/qwen2vl_language/R1/global_rotation.pth"

cal_config = get_calibration_config(output=1)
bit_config = get_bit_config(activation_16bits=["model_merger_fc2"])
quantization_config = get_quantization_config(cal_config, bit_config)

et_config = get_equivalent_transformation_config(
                head_out_ch_rotation_apply = True,
                head_out_ch_rotation_matrix_path = head_out_ch_rotation_matrix_path,
            )

advanced_config = get_advanced_quantization_config(equivalent_transformation = et_config)

mxq_compile(
    mblt_path,
    save_path=save_path,
    calib_data_path=calib_data_path,
    device=device,
    inference_scheme='multi',
    quantization=quantization_config,
    advanced_quantization=advanced_config,
)
