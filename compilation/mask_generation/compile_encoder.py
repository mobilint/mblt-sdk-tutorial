from argparse import ArgumentParser
from pathlib import Path

import sam2_host as sam
import torch
from qbcompiler import mblt_compile, mxq_compile
from qbcompiler.configs import CompileConfig
from qbcompiler.model_dict.parser.backend.torch.input_capture import (
    capture_forward_inputs,
)
from qbcompiler.model_dict.parser.patcher.parts import prepare_part

BASE_DIR = Path(__file__).resolve().parent
TARGET_DEVICES = ("aries-rb", "regulus-rb")


def capture_feed_dict(predictor, image) -> dict:
    target = prepare_part(predictor.model, "encoder")
    with capture_forward_inputs(target, to_cpu=False) as feed_dict:
        predictor.set_image(image)
    return dict(feed_dict)


def compile_encoder(target_device: str, model_id: str, image_path: str, device: str) -> Path:
    calibration_path = BASE_DIR / "calib" / "encoder" / "encoder_calib.txt"
    mblt_path = BASE_DIR / "mblt" / target_device / "sam2_hiera_large_encoder.mblt"
    mxq_path = BASE_DIR / "mxq" / target_device / "sam2_hiera_large_encoder.mxq"

    if not calibration_path.is_file():
        raise FileNotFoundError(f"Encoder calibration not found: {calibration_path}")

    torch_device = sam.resolve_device(device)
    predictor = sam.build_predictor(model_id, torch_device)
    image = sam.load_image_np(image_path)
    feed_dict = capture_feed_dict(predictor, image)

    mblt_path.parent.mkdir(parents=True, exist_ok=True)
    mblt_compile(
        model=predictor.model,
        model_part="encoder",
        backend="torch",
        target_device=target_device,
        mblt_save_path=str(mblt_path),
        feed_dict=feed_dict,
    )

    mxq_path.parent.mkdir(parents=True, exist_ok=True)
    mxq_compile(
        model=str(mblt_path),
        target_device=target_device,
        calib_data_path=str(calibration_path),
        save_path=str(mxq_path),
        device="gpu" if torch.cuda.is_available() else "cpu",
        inference_scheme="single",
        compile_config=CompileConfig.from_file(str(BASE_DIR / "compile_config.json")),
    )

    print(f"Saved encoder MXQ to {mxq_path}")
    return mxq_path


if __name__ == "__main__":
    parser = ArgumentParser(description="Compile the SAM2 image encoder")
    parser.add_argument("--target-device", choices=TARGET_DEVICES, default="aries-rb")
    parser.add_argument("--model-id", default=sam.DEFAULT_MODEL_ID)
    parser.add_argument("--image", default=str(sam.DEFAULT_IMAGE))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    compile_encoder(args.target_device, args.model_id, args.image, args.device)
