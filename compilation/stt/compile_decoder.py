"""Compile Whisper decoder to MXQ format."""

import os
from argparse import ArgumentParser

import torch
from qbcompiler import mblt_compile, mxq_compile
from qbcompiler.configs import CalibrationConfig, EquivalentTransformationConfig, LlmConfig
from transformers import AutoModelForSpeechSeq2Seq

# Paths
CALIB_PATH = "./calibration_data/decoder/whisper_decoder_calib.json"
MBLT_PATH = "./mblt/whisper-small_decoder.mblt"
MXQ_PATH = "./mxq/whisper-small_decoder.mxq"


def get_device_inference_scheme(target_device):
    # REGULUS only supports the single scheme; ARIES supports all schemes in one model.
    if "regulus" in target_device:
        return "single"
    elif "aries" in target_device:
        return "all"
    raise ValueError(f"{target_device} not supported in current qbcompiler version")


def compile_decoder(calib_path, target_device="aries-rb", mblt_path=MBLT_PATH, mxq_path=MXQ_PATH):
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
        target_device=target_device,
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

    # Conservative clip (percentile=0.999999) preserves decoder activation outliers.
    calibration_config = CalibrationConfig(
        output=0,  # per-layer output quantization
        mode=1,  # MaxPercentile
        max_percentile=CalibrationConfig.MaxPercentile(percentile=0.999999),
    )

    # Equivalent transformations for the Whisper decoder.
    etc_config = EquivalentTransformationConfig(
        qk=EquivalentTransformationConfig.Qk(apply=True),
        ud=EquivalentTransformationConfig.Ud(apply=True),
        vo=EquivalentTransformationConfig.Vo(apply=True),
        spin_r2=EquivalentTransformationConfig.SpinR2(apply=True),
        qk_rotation=EquivalentTransformationConfig.QkRotation(apply=True),
        optimize_ffn=EquivalentTransformationConfig.OptimizeFfn(apply=True, ch_per_ffn=-1),
    )

    mxq_compile(
        model=mblt_path,
        target_device=target_device,
        calib_data_path=calib_path,
        save_path=mxq_path,
        device=device,
        inference_scheme=get_device_inference_scheme(target_device),
        calibration_config=calibration_config,
        equivalent_transformation_config=etc_config,
        llm_config=llm_config,
    )
    print(f"Decoder compiled: {mxq_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Compile Whisper decoder to MXQ / MBLT")
    parser.add_argument(
        "--target-device",
        type=str,
        choices=["regulus-ra", "regulus-rb", "aries-rb"],
        default="aries-rb",
        help="Target NPU (e.g. aries-rb, regulus-rb)",
    )
    args = parser.parse_args()

    if not os.path.exists(CALIB_PATH):
        print(f"Decoder calibration not found: {CALIB_PATH}")
        print("Please run generate_calibration.py first!")
    else:
        compile_decoder(CALIB_PATH, target_device=args.target_device)
