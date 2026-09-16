from qbcompiler.artifact.input import read_mblt_graph_info
from qbcompiler.configs import (
    BitConfig,
    CalibrationConfig,
    EquivalentTransformationConfig,
    HessianQuantConfig,
    LlmConfig,
    ResourceManagementConfig,
    SearchWeightScaleConfig,
)

def get_graph_layer_names(mblt_path: str, boundary: str) -> list[str]:
    """Return the Input (boundary="inputs") or Output (boundary="outputs") layer names of an MBLT."""
    layer_type = {"inputs": "Input", "outputs": "Output"}[boundary]
    _, subgraphs = read_mblt_graph_info(mblt_path)
    names: list[str] = []
    for subgraph in subgraphs:
        producers = {}  # activation id -> boundary layer name
        for op in subgraph.operators:
            op_type = getattr(op.layertype, "name", str(op.layertype)).rsplit(".", 1)[-1]
            if op_type == layer_type:
                for activation_id in op.options.outputs:
                    producers[int(activation_id)] = op.name
        for activation_id in getattr(subgraph, boundary):
            name = producers.get(int(activation_id))
            if name is not None and name not in names:
                names.append(name)
    if not names:
        raise RuntimeError(f"no {layer_type} layers found in {mblt_path}")
    return names


def activation_16bit_config(mblt_path: str, boundary: str) -> BitConfig:
    """Keep the MBLT graph inputs (decoder) or outputs (encoder) in 16-bit.

    Layer names are read from the MBLT because the parser names them per model size
    and per trace path. The dynamic decoder RoPE input is added later by the quantizer,
    so it is not part of the MBLT inputs.
    """
    return BitConfig(
        layer_overrides=BitConfig.LayerOverrides(activation_16bits=get_graph_layer_names(mblt_path, boundary)),
    )


def spin_rotation_relpath(target_device: str, model_name: str, dynamic: bool) -> str:
    subdir = f"{model_name}-dynamic" if dynamic else model_name
    return f"spinWeight/{target_device}/{subdir}/global_rotation.pth"


def _llm_runtime(dynamic: bool) -> LlmConfig.Attributes.Runtime:
    return LlmConfig.Attributes.Runtime(dynamic_rope=dynamic)


def decoder_compile_config(target_device: str, mblt_path: str, dynamic: bool = False) -> dict:
    bit_config = BitConfig(
        transformer=BitConfig.Transformer(
            weight=BitConfig.Transformer.Weight(
                query=4,
                key=4,
                value=8,
                output=4,
                ffn=4,
                head=4,
            ),
        ),
        layer_overrides=BitConfig.LayerOverrides(activation_16bits=get_graph_layer_names(mblt_path, "inputs")),
    )
    if target_device == "regulus-rb":
        return {
            "inference_scheme": "single",
            "calibration_config": CalibrationConfig(output=0, mode=0),
            "bit_config": bit_config,
            "resource_management_config": ResourceManagementConfig(
                weight_dtype="float32",
                use_gpu_only_for_calibration=True,
                weight_memory=ResourceManagementConfig.WeightMemory(method=1),
            ),
            "llm_config": LlmConfig(
                apply=True,
                attributes=LlmConfig.Attributes(
                    max_sequence_length=4096,
                    max_cache_length=4096,
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
                accumulation_device="gpu",
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
            "bit_config": bit_config,
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
            "hessian_quant_config": HessianQuantConfig(
                apply=True,
                accumulation_device="gpu",
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
    else:
        raise ValueError(f"Unsupported target device: {target_device}")


def encoder_compile_config(
    target_device: str,
    model_name: str,
    mblt_path: str,
    dynamic: bool = False,
) -> dict:
    rotation_matrix_path = spin_rotation_relpath(target_device, model_name, dynamic)
    bit_config = activation_16bit_config(mblt_path, "outputs")
    if target_device == "regulus-rb":
        return {
            "inference_scheme": "single",
            "calibration_config": CalibrationConfig(output=0, mode=1),
            "bit_config": bit_config,
            "resource_management_config": ResourceManagementConfig(
                weight_dtype="float32",
                use_gpu_only_for_calibration=True,
                weight_memory=ResourceManagementConfig.WeightMemory(method=1),
            ),
            "equivalent_transformation_config": EquivalentTransformationConfig(
                qk=EquivalentTransformationConfig.Qk(apply=True),
                ud=EquivalentTransformationConfig.Ud(apply=True),
                vo=EquivalentTransformationConfig.Vo(apply=True),
                head_out_ch_rotation=EquivalentTransformationConfig.HeadOutChRotation(
                    apply=True,
                    matrix_path=rotation_matrix_path,
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
            "bit_config": bit_config,
            "resource_management_config": ResourceManagementConfig(
                weight_dtype="float32",
                use_gpu_only_for_calibration=True,
                weight_memory=ResourceManagementConfig.WeightMemory(method=1),
            ),
            "equivalent_transformation_config": EquivalentTransformationConfig(
                qk=EquivalentTransformationConfig.Qk(apply=True),
                ud=EquivalentTransformationConfig.Ud(apply=True),
                vo=EquivalentTransformationConfig.Vo(apply=True),
                head_out_ch_rotation=EquivalentTransformationConfig.HeadOutChRotation(
                    apply=True,
                    matrix_path=rotation_matrix_path,
                ),
                spin_r1=EquivalentTransformationConfig.SpinR1(apply=False),
                spin_r2=EquivalentTransformationConfig.SpinR2(apply=True),
                optimize_ffn=EquivalentTransformationConfig.OptimizeFfn(apply=True),
            ),
        }
    else:
        raise ValueError(f"Unsupported target device: {target_device}")
