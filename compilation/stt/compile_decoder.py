"""Compile Whisper decoder to MXQ format."""

import os

import torch
from qbcompiler import mblt_compile, mxq_compile
from qbcompiler.configs import LlmConfig
from transformers import AutoModelForSpeechSeq2Seq

# Paths
CALIB_PATH = "./calibration_data/decoder/whisper_decoder_calib.json"
MBLT_PATH = "./mblt/whisper-small_decoder.mblt"
MXQ_PATH = "./mxq/whisper-small_decoder.mxq"


def compile_decoder(calib_path, mblt_path=MBLT_PATH, mxq_path=MXQ_PATH):
    """Compile Whisper decoder to MXQ."""

    os.makedirs(os.path.dirname(mblt_path), exist_ok=True)
    os.makedirs(os.path.dirname(mxq_path), exist_ok=True)

    # Load model
    print("Loading Whisper model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small")
    model = model.eval().cpu()

    # MBLT compile
    print(f"Compiling to MBLT: {mblt_path}")
    mblt_compile(
        model=model,
        mblt_save_path=mblt_path,
        backend="hf",
        target="decoder",
        device="cpu",
    )

    # MXQ compile
    device = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Compiling to MXQ: {mxq_path} (device: {device})")
    llm_config = LlmConfig(
        apply=True,
        attributes=LlmConfig.Attributes(
            calibration=LlmConfig.Attributes.Calibration(use_full_seq_length=True),
        ),
    )
    mxq_compile(
        model=mblt_path,
        calib_data_path=calib_path,
        save_path=mxq_path,
        device=device,
        inference_scheme="single",
        llm_config=llm_config,
    )
    print(f"Decoder compiled: {mxq_path}")


if __name__ == "__main__":
    if not os.path.exists(CALIB_PATH):
        print(f"Decoder calibration not found: {CALIB_PATH}")
        print("Please run generate_calibration.py first!")
    else:
        compile_decoder(CALIB_PATH)
