from argparse import ArgumentParser
from pathlib import Path

import torch
from compile_config import (
    bit_config,
    equivalent_transformation_config,
    inference_scheme,
)
from qbcompiler import mblt_compile, mxq_compile
from transformers import AutoModelForSpeechSeq2Seq

MODEL_ID = "openai/whisper-small"
TARGET_DEVICES = ("aries-rb", "regulus-rb")


def compile_encoder(target_device: str) -> Path:
    calibration_path = Path("calibration_data/encoder/whisper_encoder_cali.txt")
    mblt_path = Path("mblt") / target_device / "whisper-small_encoder.mblt"
    mxq_path = Path("mxq") / target_device / "whisper-small_encoder.mxq"

    if not calibration_path.is_file():
        raise FileNotFoundError(f"Encoder calibration not found: {calibration_path}")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval()
    model.cpu()
    feed_dict = {"input_features": torch.randn(1, 80, 3000)}

    mblt_path.parent.mkdir(parents=True, exist_ok=True)
    mblt_compile(
        model=model,
        mblt_save_path=str(mblt_path),
        target_device=target_device,
        backend="hf",
        target="encoder",
        feed_dict=feed_dict,
        device="cpu",
    )

    mxq_path.parent.mkdir(parents=True, exist_ok=True)
    mxq_compile(
        model=str(mblt_path),
        target_device=target_device,
        calib_data_path=str(calibration_path),
        save_path=str(mxq_path),
        device="gpu" if torch.cuda.is_available() else "cpu",
        inference_scheme=inference_scheme(target_device),
        equivalent_transformation_config=equivalent_transformation_config(),
        bit_config=bit_config(),
    )

    print(f"Saved encoder MXQ to {mxq_path}")
    return mxq_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--target-device", choices=TARGET_DEVICES, default="aries-rb")
    args = parser.parse_args()

    compile_encoder(args.target_device)
