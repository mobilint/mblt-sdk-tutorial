from qbcompiler.configs import (
    BitConfig,
    CalibrationConfig,
    EquivalentTransformationConfig,
    HessianQuantConfig,
    LlmConfig,
    ResourceManagementConfig,
    SearchWeightScaleConfig,
)

TARGET_DEVICES = ("aries-rb", "regulus-rb")
DECODER_INPUT_NAMES = [
    "inputs_embeds/reshape",
    "deepstack_visual_embeds/reshape/slice",
    "deepstack_visual_embeds/reshape/slice_0",
    "deepstack_visual_embeds/reshape/slice_1",
]
ENCODER_16BIT_ACTIVATIONS = [
    "model_merger_fc2_conv_channel_last",
    "add/reshape_49/reshape/gelu/conv2d",
    "add/reshape_99/reshape/gelu/conv2d",
    "add/reshape_149/reshape/gelu/conv2d",
]


def decoder_compile_config(target_device: str) -> dict:
    if target_device == "regulus-rb":
        return {
            "inference_scheme": "single",
            "calibration_config": CalibrationConfig(output=0, mode=0),
            "bit_config": BitConfig(
                layer_overrides=BitConfig.LayerOverrides(activation_16bits=DECODER_INPUT_NAMES),
            ),
            "resource_management_config": ResourceManagementConfig(
                weight_dtype="float32",
                use_gpu_only_for_calibration=True,
                weight_memory=ResourceManagementConfig.WeightMemory(method=1),
            ),
            "llm_config": LlmConfig(
                apply=True,
                attributes=LlmConfig.Attributes(
                    max_sequence_length=1024,
                    max_cache_length=1024,
                    calibration=LlmConfig.Attributes.Calibration(use_full_seq_length=True),
                ),
            ),
            "equivalent_transformation_config": EquivalentTransformationConfig(
                qk=EquivalentTransformationConfig.Qk(apply=False),
                ud=EquivalentTransformationConfig.Ud(apply=True, smoothing_factor=0.8),
                vo=EquivalentTransformationConfig.Vo(apply=True),
                spin_r1=EquivalentTransformationConfig.SpinR1(apply=True),
                spin_r2=EquivalentTransformationConfig.SpinR2(apply=True),
                optimize_ffn=EquivalentTransformationConfig.OptimizeFfn(apply=True),
            ),
            "hessian_quant_config": HessianQuantConfig(
                apply=True,
                attributes=HessianQuantConfig.Attributes(
                    act_order=True,
                    block_size=128,
                    perc_damp=0.01,
                ),
            ),
            "search_weight_scale_config": SearchWeightScaleConfig(
                apply=True,
                transformer=SearchWeightScaleConfig.Transformer(
                    query=True,
                    key=True,
                    value=True,
                    out=True,
                    ffn=True,
                ),
            ),
        }
    elif target_device == "aries-rb":
        return {
            "inference_scheme": "all",
            "calibration_config": CalibrationConfig(output=0, mode=0),
            "bit_config": BitConfig(
                layer_overrides=BitConfig.LayerOverrides(activation_16bits=DECODER_INPUT_NAMES),
            ),
            "resource_management_config": ResourceManagementConfig(
                weight_dtype="float32",
                use_gpu_only_for_calibration=True,
                weight_memory=ResourceManagementConfig.WeightMemory(method=1),
            ),
            "llm_config": LlmConfig(
                apply=True,
                attributes=LlmConfig.Attributes(
                    calibration=LlmConfig.Attributes.Calibration(use_full_seq_length=True),
                ),
            ),
            "equivalent_transformation_config": EquivalentTransformationConfig(
                qk=EquivalentTransformationConfig.Qk(apply=False),
                ud=EquivalentTransformationConfig.Ud(apply=True, smoothing_factor=0.8),
                vo=EquivalentTransformationConfig.Vo(apply=True),
                spin_r1=EquivalentTransformationConfig.SpinR1(apply=True),
                spin_r2=EquivalentTransformationConfig.SpinR2(apply=True),
                optimize_ffn=EquivalentTransformationConfig.OptimizeFfn(apply=True),
            ),
            "hessian_quant_config": None,
            "search_weight_scale_config": SearchWeightScaleConfig(
                apply=True,
                transformer=SearchWeightScaleConfig.Transformer(
                    query=True,
                    key=True,
                    value=True,
                    out=True,
                    ffn=True,
                ),
            ),
        }
    else:
        raise ValueError(f"Unsupported target device: {target_device}")


def encoder_compile_config(target_device: str) -> dict:
    if target_device == "regulus-rb":
        return {
            "inference_scheme": "single",
            "calibration_config": CalibrationConfig(output=0, mode=1),
            "bit_config": BitConfig(
                layer_overrides=BitConfig.LayerOverrides(activation_16bits=ENCODER_16BIT_ACTIVATIONS),
            ),
            "resource_management_config": ResourceManagementConfig(
                weight_dtype="float32",
                use_gpu_only_for_calibration=True,
                weight_memory=ResourceManagementConfig.WeightMemory(method=1),
            ),
            "llm_config": LlmConfig(
                apply=True,
                attributes=LlmConfig.Attributes(
                    max_sequence_length=1024,
                    max_cache_length=1024,
                    calibration=LlmConfig.Attributes.Calibration(use_full_seq_length=True),
                ),
            ),
            "equivalent_transformation_config": EquivalentTransformationConfig(
                qk=EquivalentTransformationConfig.Qk(apply=True),
                ud=EquivalentTransformationConfig.Ud(apply=True),
                vo=EquivalentTransformationConfig.Vo(apply=True),
                head_out_ch_rotation=EquivalentTransformationConfig.HeadOutChRotation(
                    apply=True,
                    matrix_path="spinWeight/regulus-rb/global_rotation.pth",
                ),
                spin_r1=EquivalentTransformationConfig.SpinR1(apply=False),
                spin_r2=EquivalentTransformationConfig.SpinR2(apply=True),
                optimize_ffn=EquivalentTransformationConfig.OptimizeFfn(apply=True),
            ),
        }
    elif target_device == "aries-rb":
        return {
            "inference_scheme": "all",
            "calibration_config": CalibrationConfig(output=0, mode=1),
            "bit_config": BitConfig(
                layer_overrides=BitConfig.LayerOverrides(activation_16bits=ENCODER_16BIT_ACTIVATIONS),
            ),
            "resource_management_config": ResourceManagementConfig(
                weight_dtype="float32",
                use_gpu_only_for_calibration=True,
                weight_memory=ResourceManagementConfig.WeightMemory(method=1),
            ),
            "llm_config": LlmConfig(
                apply=True,
                attributes=LlmConfig.Attributes(
                    calibration=LlmConfig.Attributes.Calibration(use_full_seq_length=True),
                ),
            ),
            "equivalent_transformation_config": EquivalentTransformationConfig(
                qk=EquivalentTransformationConfig.Qk(apply=True),
                ud=EquivalentTransformationConfig.Ud(apply=True),
                vo=EquivalentTransformationConfig.Vo(apply=True),
                head_out_ch_rotation=EquivalentTransformationConfig.HeadOutChRotation(
                    apply=True,
                    matrix_path="spinWeight/aries-rb/global_rotation.pth",
                ),
                spin_r1=EquivalentTransformationConfig.SpinR1(apply=False),
                spin_r2=EquivalentTransformationConfig.SpinR2(apply=True),
                optimize_ffn=EquivalentTransformationConfig.OptimizeFfn(apply=True),
            ),
        }
    else:
        raise ValueError(f"Unsupported target device: {target_device}")
