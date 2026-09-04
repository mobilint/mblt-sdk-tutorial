from qbcompiler.configs import (
    BitConfig,
    CalibrationConfig,
    EquivalentTransformationConfig,
    HessianQuantConfig,
    LlmConfig,
)


def decoder_compile_config(target_device: str) -> dict:
    if target_device == "regulus-rb":
        return {
            "inference_scheme": "single",
            "calibration_config": CalibrationConfig(output=0, mode=0),
            "equivalent_transformation_config": EquivalentTransformationConfig(
                norm_conv=EquivalentTransformationConfig.NormConv(apply=True, learn=True),
                qk=EquivalentTransformationConfig.Qk(apply=True),
                ud=EquivalentTransformationConfig.Ud(apply=True, learn=True),
                vo=EquivalentTransformationConfig.Vo(apply=True),
                feed_forward_multi_lut=EquivalentTransformationConfig.FeedForwardMultiLut(apply=True),
                spin_r2=EquivalentTransformationConfig.SpinR2(apply=True, learn=True),
                qk_rotation=EquivalentTransformationConfig.QkRotation(apply=True),
                flatten_quant=EquivalentTransformationConfig.FlattenQuant(apply=True, learn=True),
                optimize_ffn=EquivalentTransformationConfig.OptimizeFfn(apply=True, ch_per_ffn=-1),
            ),
            "llm_config": LlmConfig(
                apply=True,
                attributes=LlmConfig.Attributes(
                    max_sequence_length=1024,
                    max_cache_length=1024,
                    calibration=LlmConfig.Attributes.Calibration(use_full_seq_length=True),
                ),
            ),
            "bit_config": BitConfig(
                transformer=BitConfig.Transformer(
                    mixed_precision=BitConfig.Transformer.MixedPrecision(
                        activation=BitConfig.Transformer.MixedPrecision.Activation(apply=True),
                    ),
                ),
            ),
            "hessian_quant_config": HessianQuantConfig(
                apply=True,
                attributes=HessianQuantConfig.Attributes(
                    actOrder=True,
                    blockSize=128,
                    percDamp=0.01,
                ),
            ),
        }
    elif target_device == "aries-rb":
        return {
            "inference_scheme": "all",
            "calibration_config": CalibrationConfig(output=0, mode=0),
            "equivalent_transformation_config": EquivalentTransformationConfig(
                norm_conv=EquivalentTransformationConfig.NormConv(apply=True, learn=True),
                qk=EquivalentTransformationConfig.Qk(apply=True),
                ud=EquivalentTransformationConfig.Ud(apply=True, learn=True),
                vo=EquivalentTransformationConfig.Vo(apply=True),
                feed_forward_multi_lut=EquivalentTransformationConfig.FeedForwardMultiLut(apply=True),
                spin_r2=EquivalentTransformationConfig.SpinR2(apply=True, learn=True),
                qk_rotation=EquivalentTransformationConfig.QkRotation(apply=True),
                flatten_quant=EquivalentTransformationConfig.FlattenQuant(apply=True, learn=True),
                optimize_ffn=EquivalentTransformationConfig.OptimizeFfn(apply=True, ch_per_ffn=-1),
            ),
            "llm_config": LlmConfig(
                apply=True,
                attributes=LlmConfig.Attributes(
                    calibration=LlmConfig.Attributes.Calibration(use_full_seq_length=True),
                ),
            ),
            "bit_config": BitConfig(
                transformer=BitConfig.Transformer(
                    mixed_precision=BitConfig.Transformer.MixedPrecision(
                        activation=BitConfig.Transformer.MixedPrecision.Activation(apply=True),
                    ),
                ),
            ),
            "hessian_quant_config": HessianQuantConfig(
                apply=True,
                attributes=HessianQuantConfig.Attributes(
                    actOrder=True,
                    blockSize=128,
                    percDamp=0.01,
                ),
            ),
        }
    else:
        raise ValueError(f"Unsupported target device: {target_device}")


def encoder_compile_config(target_device: str) -> dict:
    if target_device == "regulus-rb":
        return {
            "inference_scheme": "single",
            "equivalent_transformation_config": EquivalentTransformationConfig(
                norm_conv=EquivalentTransformationConfig.NormConv(apply=True, learn=True),
                qk=EquivalentTransformationConfig.Qk(apply=True),
                ud=EquivalentTransformationConfig.Ud(apply=True, learn=True),
                vo=EquivalentTransformationConfig.Vo(apply=True),
                feed_forward_multi_lut=EquivalentTransformationConfig.FeedForwardMultiLut(apply=True),
                spin_r2=EquivalentTransformationConfig.SpinR2(apply=True, learn=True),
                qk_rotation=EquivalentTransformationConfig.QkRotation(apply=True),
                flatten_quant=EquivalentTransformationConfig.FlattenQuant(apply=True, learn=True),
                optimize_ffn=EquivalentTransformationConfig.OptimizeFfn(apply=True, ch_per_ffn=-1),
            ),
            "bit_config": BitConfig(
                transformer=BitConfig.Transformer(
                    mixed_precision=BitConfig.Transformer.MixedPrecision(
                        activation=BitConfig.Transformer.MixedPrecision.Activation(apply=True),
                    ),
                ),
            ),
        }
    elif target_device == "aries-rb":
        return {
            "inference_scheme": "all",
            "equivalent_transformation_config": EquivalentTransformationConfig(
                norm_conv=EquivalentTransformationConfig.NormConv(apply=True, learn=True),
                qk=EquivalentTransformationConfig.Qk(apply=True),
                ud=EquivalentTransformationConfig.Ud(apply=True, learn=True),
                vo=EquivalentTransformationConfig.Vo(apply=True),
                feed_forward_multi_lut=EquivalentTransformationConfig.FeedForwardMultiLut(apply=True),
                spin_r2=EquivalentTransformationConfig.SpinR2(apply=True, learn=True),
                qk_rotation=EquivalentTransformationConfig.QkRotation(apply=True),
                flatten_quant=EquivalentTransformationConfig.FlattenQuant(apply=True, learn=True),
                optimize_ffn=EquivalentTransformationConfig.OptimizeFfn(apply=True, ch_per_ffn=-1),
            ),
            "bit_config": BitConfig(
                transformer=BitConfig.Transformer(
                    mixed_precision=BitConfig.Transformer.MixedPrecision(
                        activation=BitConfig.Transformer.MixedPrecision.Activation(apply=True),
                    ),
                ),
            ),
        }
    else:
        raise ValueError(f"Unsupported target device: {target_device}")
