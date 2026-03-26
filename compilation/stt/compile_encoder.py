"""Compile Whisper encoder to MXQ format."""

import os

import torch
from qbcompiler import mblt_compile, mxq_compile
from transformers import AutoModelForSpeechSeq2Seq

# Paths
CALIB_PATH = "./calibration_data/encoder/whisper_encoder_cali.txt"
MBLT_PATH = "./mblt/whisper-small_encoder.mblt"
MXQ_PATH = "./mxq/whisper-small_encoder.mxq"


def compile_encoder(calib_path, mblt_path=MBLT_PATH, mxq_path=MXQ_PATH):
    """Compile Whisper encoder to MXQ."""

    os.makedirs(os.path.dirname(mblt_path), exist_ok=True)
    os.makedirs(os.path.dirname(mxq_path), exist_ok=True)

    # Load model
    print("Loading Whisper model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small")
    model = model.eval().cpu()

    # Sample input for tracing: mel spectrogram [1, 80, 3000]
    feed_dict = {"input_features": torch.randn(1, 80, 3000)}

    # MBLT compile
    print(f"Compiling to MBLT: {mblt_path}")
    mblt_compile(
        model=model,
        mblt_save_path=mblt_path,
        backend="hf",
        target="encoder",
        feed_dict=feed_dict,
        device="cpu",
    )

    # MXQ compile
    device = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Compiling to MXQ: {mxq_path} (device: {device})")
    mxq_compile(
        model=mblt_path,
        calib_data_path=calib_path,
        save_path=mxq_path,
        device=device,
        inference_scheme="all",
    )
    print(f"Encoder compiled: {mxq_path}")


if __name__ == "__main__":
    if not os.path.exists(CALIB_PATH):
        print(f"Encoder calibration not found: {CALIB_PATH}")
        print("Please run generate_calibration.py first!")
    else:
        compile_encoder(CALIB_PATH)
