from qbcompiler.configs import (
    BitConfig,
    CalibrationConfig,
    EquivalentTransformationConfig,
    HessianQuantConfig,
    LlmConfig,
)


def inference_scheme(target_device: str) -> str:
    if target_device == "aries-rb":
        return "all"
    if target_device == "regulus-rb":
        return "single"
    raise ValueError(f"Unsupported target device: {target_device}")


def equivalent_transformation_config() -> EquivalentTransformationConfig:
    return EquivalentTransformationConfig(
        norm_conv=EquivalentTransformationConfig.NormConv(apply=True, learn=True),
        qk=EquivalentTransformationConfig.Qk(apply=True),
        ud=EquivalentTransformationConfig.Ud(apply=True, learn=True),
        vo=EquivalentTransformationConfig.Vo(apply=True),
        feed_forward_multi_lut=EquivalentTransformationConfig.FeedForwardMultiLut(apply=True),
        spin_r2=EquivalentTransformationConfig.SpinR2(apply=True, learn=True),
        qk_rotation=EquivalentTransformationConfig.QkRotation(apply=True),
        flatten_quant=EquivalentTransformationConfig.FlattenQuant(apply=True, learn=True),
        optimize_ffn=EquivalentTransformationConfig.OptimizeFfn(apply=True, ch_per_ffn=-1),
    )


def bit_config() -> BitConfig:
    return BitConfig(
        transformer=BitConfig.Transformer(
            mixed_precision=BitConfig.Transformer.MixedPrecision(
                activation=BitConfig.Transformer.MixedPrecision.Activation(apply=True),
            ),
        ),
    )


def decoder_calibration_config() -> CalibrationConfig:
    return CalibrationConfig(output=0, mode=0)


def decoder_llm_config(target_device: str) -> LlmConfig:
    calibration = LlmConfig.Attributes.Calibration(use_full_seq_length=True)
    if target_device == "aries-rb":
        attributes = LlmConfig.Attributes(calibration=calibration)
    elif target_device == "regulus-rb":
        attributes = LlmConfig.Attributes(
            max_sequence_length=1024,
            max_cache_length=1024,
            calibration=calibration,
        )
    else:
        raise ValueError(f"Unsupported target device: {target_device}")
    return LlmConfig(apply=True, attributes=attributes)


def hessian_quant_config() -> HessianQuantConfig:
    return HessianQuantConfig(
        apply=True,
        attributes=HessianQuantConfig.Attributes(
            actOrder=True,
            blockSize=128,
            percDamp=0.01,
        ),
    )
