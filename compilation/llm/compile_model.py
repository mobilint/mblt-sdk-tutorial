from argparse import ArgumentParser
from pathlib import Path

import torch
from qbcompiler import (
    BitConfig,
    CalibrationConfig,
    EquivalentTransformationConfig,
    LlmConfig,
    mblt_compile,
    mxq_compile,
)
from qbcompiler.model_dict.parser.backend.torch.input_capture import (
    capture_forward_inputs,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device_inference_scheme(target_device: str) -> str:
    if "regulus" in target_device:
        return "single"
    if "aries" in target_device:
        return "all"
    raise ValueError(f"{target_device} not supported in current qbcompiler version")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-path", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument(
        "--mblt-path",
        type=Path,
        default=Path("./Llama-3.2-1B-Instruct.mblt"),
    )
    parser.add_argument(
        "--calib-data-path",
        type=str,
        default="./calibration_data/datas/meta-llama-Llama-3.2-1B-Instruct/multilingual",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("./Llama-3.2-1B-Instruct-W8.mxq"),
    )
    parser.add_argument(
        "--target-device",
        type=str,
        choices=["regulus-rb", "aries-rb"],
        default="aries-rb",
        help="Target NPU (e.g. aries-rb, regulus-rb)",
    )
    args = parser.parse_args()

    device = "gpu" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    inputs = tokenizer("Hello", return_tensors="pt")

    with capture_forward_inputs(model) as feed_dict:
        model.generate(**inputs, max_new_tokens=1, do_sample=False)

    feed_dict["inputs_embeds"] = model.get_input_embeddings()(feed_dict["input_ids"]).detach()
    feed_dict["input_ids"] = None
    feed_dict["attention_mask"] = None

    args.mblt_path.parent.mkdir(parents=True, exist_ok=True)
    mblt_compile(
        model=model,
        backend="torch",
        target_device=args.target_device,
        mblt_save_path=str(args.mblt_path),
        feed_dict=dict(feed_dict),
        dynamic_axes={"inputs_embeds": [-2]},
    )

    if args.target_device == "aries-rb":
        max_sequence_length = 4096
        max_cache_length = 4096
    elif args.target_device == "regulus-rb":
        max_sequence_length = 1024
        max_cache_length = 1024
    else:
        raise ValueError(f"not support {args.target_device}")

    calib_config = CalibrationConfig(
        method=1,
        output=0,
        mode=1,
    )

    bit_config = BitConfig(
        transformer=BitConfig.Transformer(
            weight=BitConfig.Transformer.Weight(
                query=8,
                key=8,
                value=8,
                output=8,
                ffn=8,
                head=8,
            ),
        )
    )

    llm_config = LlmConfig(
        apply=True,
        attributes=LlmConfig.Attributes(
            max_sequence_length=max_sequence_length,
            max_cache_length=max_cache_length,
            max_core_data_length=128,
            calibration=LlmConfig.Attributes.Calibration(use_full_seq_length=True),
            runtime=LlmConfig.Attributes.Runtime(batch_size=1, npu_core_ids=[0]),
        ),
    )

    et_config = EquivalentTransformationConfig(
        spin_r1=EquivalentTransformationConfig.SpinR1(apply=True),
    )

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    mxq_compile(
        model=str(args.mblt_path),
        target_device=args.target_device,
        calib_data_path=args.calib_data_path,
        save_path=str(args.save_path),
        device=device,
        inference_scheme=get_device_inference_scheme(args.target_device),
        calibration_config=calib_config,
        bit_config=bit_config,
        llm_config=llm_config,
        equivalent_transformation_config=et_config,
    )

    print(f"Saved MBLT model to {args.mblt_path}")
    print(f"Saved MXQ model to {args.save_path}")
