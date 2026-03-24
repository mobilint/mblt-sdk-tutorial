#!/usr/bin/env python3
"""
Compile Whisper Encoder to MXQ format
Separated from decoder compilation for efficiency
"""

import os

import torch
from qbcompiler import mblt_compile, mxq_compile
from transformers import AutoModelForSpeechSeq2Seq


def compile_encoder(calib_path, output_dir="./compiled"):
    """Compile Whisper encoder to MXQ"""

    print("=" * 60)
    print("🎵 Compiling Whisper Encoder")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Load Whisper model
    print("Loading Whisper model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-small", dtype=torch.float32, low_cpu_mem_usage=True
    )
    model = model.eval().cpu()

    mblt_path = os.path.join(output_dir, "whisper-small_encoder.mblt")
    mblt_compile(
        model=model,
        mblt_save_path=mblt_path,
        backend="torch",
        target="encoder",
        device="cpu",
    )

    # Compile to MXQ
    print("Compiling to MXQ...")
    mxq_path = os.path.join(output_dir, "whisper-small_encoder.mxq")

    try:
        mxq_compile(
            model=mblt_path,
            calib_data_path=calib_path,
            save_path=mxq_path,
            device="cuda",
            inference_scheme="global4",
        )

        print(f"✅ Encoder compiled: {mxq_path}")
        return mxq_path

    except Exception as e:
        print(f"❌ Encoder compilation failed: {e}")
        return None


def main():
    """Compile Whisper encoder"""

    print("🚀 Compiling Whisper Encoder to MXQ")
    print("=" * 70)

    # Check calibration data
    encoder_calib_path = "../calibration/encoder/whisper_encoder_cali.txt"

    if not os.path.exists(encoder_calib_path):
        print(f"❌ Encoder calibration not found: {encoder_calib_path}")
        print("Please run create_calibration.py first!")
        return 1

    # Compile encoder
    encoder_mxq_path = compile_encoder(encoder_calib_path)

    print("\n" + "=" * 70)
    print("🎉 Encoder Compilation Complete!")
    print("=" * 70)

    if encoder_mxq_path:
        print(f"✅ Encoder: {encoder_mxq_path}")
    else:
        print("❌ Encoder compilation failed")

    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
