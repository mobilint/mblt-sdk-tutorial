from qbcompiler.configs import (
    BitConfig,
    CalibrationConfig,
    EquivalentTransformationConfig,
    HessianQuantConfig,
    LlmConfig,
    ResourceManagementConfig,
    SearchWeightScaleConfig,
)

DECODER_16BIT_ACTIVATIONS = [
    "inputs_embeds/reshape",
    "deepstack_visual_embeds_0",
]
# Vision encoder graph-derived operator names. The three ``add/reshape_<N>/...``
# entries were observed in the static Qwen3-VL-2B .mblt; a different model size
# or the dynamic parsing path can emit different ``<N>`` values. Mismatched
# entries are silently ignored by the quantizer, so wrong names cost a small
# amount of vision SQNR but do not break the build.
ENCODER_16BIT_ACTIVATIONS = [
    "model_merger_fc2_conv_channel_last",
    "add/reshape_49/reshape/gelu/conv2d",
    "add/reshape_99/reshape/gelu/conv2d",
    "add/reshape_149/reshape/gelu/conv2d",
]


def _llm_runtime(dynamic: bool) -> LlmConfig.Attributes.Runtime:
    """Runtime knobs for the decoder LlmConfig.

    ``dynamic_rope=True`` promotes the InputConstant cos/sin tables in the
    parsed .mblt to graph inputs, turning the decoder into a 3-input MXQ that
    consumes a per-image rope tensor. The mblt-model-zoo runtime enforces a
    bundled pairing: a dynamic vision MXQ must be loaded with a dynamic text
    MXQ, so this flag flips together with the vision encoder mode.
    """
    return LlmConfig.Attributes.Runtime(dynamic_rope=dynamic)


def decoder_compile_config(target_device: str, dynamic: bool = False) -> dict:
    if target_device == "regulus-rb":
        return {
            "inference_scheme": "single",
            "calibration_config": CalibrationConfig(output=0, mode=0),
            "bit_config": BitConfig(
                layer_overrides=BitConfig.LayerOverrides(activation_16bits=DECODER_16BIT_ACTIVATIONS),
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
                    runtime=_llm_runtime(dynamic),
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
                layer_overrides=BitConfig.LayerOverrides(activation_16bits=DECODER_16BIT_ACTIVATIONS),
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
                    runtime=_llm_runtime(dynamic),
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


def encoder_compile_config(target_device: str, dynamic: bool = False) -> dict:
    # Static and dynamic parsing produce different graphs but the same
    # quantization knobs are valid for both. The activation_16bits list is a
    # best-effort optimization; see the comment on ENCODER_16BIT_ACTIVATIONS.
    del dynamic  # currently no per-mode knob divergence; kept for API symmetry
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
