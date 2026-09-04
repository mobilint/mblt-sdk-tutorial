from argparse import ArgumentParser
from pathlib import Path

import torch
from compile_config import decoder_compile_config
from qbcompiler import mblt_compile, mxq_compile
from transformers import AutoModelForSpeechSeq2Seq

MODEL_ID = "openai/whisper-small"
TARGET_DEVICES = ("aries-rb", "regulus-rb")


def compile_decoder(target_device: str) -> Path:
    calibration_path = Path("calibration_data/decoder/whisper_decoder_calib.json")
    mblt_path = Path("mblt") / target_device / "whisper-small_decoder.mblt"
    mxq_path = Path("mxq") / target_device / "whisper-small_decoder.mxq"

    if not calibration_path.is_file():
        raise FileNotFoundError(f"Decoder calibration not found: {calibration_path}")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval()
    model.cpu()

    mblt_path.parent.mkdir(parents=True, exist_ok=True)
    mblt_compile(
        model=model,
        mblt_save_path=str(mblt_path),
        target_device=target_device,
        backend="hf",
        target="decoder",
        device="cpu",
    )

    mxq_path.parent.mkdir(parents=True, exist_ok=True)
    mxq_compile(
        model=str(mblt_path),
        target_device=target_device,
        calib_data_path=str(calibration_path),
        save_path=str(mxq_path),
        device="gpu" if torch.cuda.is_available() else "cpu",
        **decoder_compile_config(target_device),
    )

    print(f"Saved decoder MXQ to {mxq_path}")
    return mxq_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--target-device", choices=TARGET_DEVICES, default="aries-rb")
    args = parser.parse_args()

    compile_decoder(args.target_device)
