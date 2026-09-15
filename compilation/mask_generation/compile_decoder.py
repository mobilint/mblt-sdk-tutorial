from argparse import ArgumentParser
from pathlib import Path

import sam2_host as sam
import torch
from prepare_calibration import write_decoder_manifest
from qbcompiler import mblt_compile, mxq_compile
from qbcompiler.configs import CompileConfig
from qbcompiler.model_dict.parser.backend.torch.input_capture import (
    capture_forward_inputs,
)
from qbcompiler.model_dict.parser.patcher.parts import prepare_part

BASE_DIR = Path(__file__).resolve().parent
TARGET_DEVICES = ("aries-rb", "regulus-rb")


def capture_feed_dict(predictor, image) -> dict:
    predictor.set_image(image)
    target = prepare_part(predictor.model, "decoder")
    points, labels = sam.prompt_arrays()
    with capture_forward_inputs(target, to_cpu=False) as feed_dict:
        predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)
    return dict(feed_dict)


def compile_decoder(target_device: str, model_id: str, image_path: str, device: str) -> Path:
    calibration_dir = BASE_DIR / "calib" / "decoder"
    mblt_path = BASE_DIR / "mblt" / target_device / "sam2_hiera_large_decoder.mblt"
    mxq_path = BASE_DIR / "mxq" / target_device / "sam2_hiera_large_decoder.mxq"

    torch_device = sam.resolve_device(device)
    predictor = sam.build_predictor(model_id, torch_device)
    image = sam.load_image_np(image_path)
    feed_dict = capture_feed_dict(predictor, image)

    mblt_path.parent.mkdir(parents=True, exist_ok=True)
    mblt_compile(
        model=predictor.model,
        model_part="decoder",
        backend="torch",
        target_device=target_device,
        mblt_save_path=str(mblt_path),
        feed_dict=feed_dict,
        dynamic_axes={"tokens": 1},
    )

    calibration_path = write_decoder_manifest(
        decoder_model=mblt_path,
        decoder_output_dir=calibration_dir,
        decoder_input_bindings=BASE_DIR / "decoder_input_bindings.json",
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

    print(f"Saved decoder MXQ to {mxq_path}")
    return mxq_path


if __name__ == "__main__":
    parser = ArgumentParser(description="Compile the SAM2 mask decoder")
    parser.add_argument("--target-device", choices=TARGET_DEVICES, default="aries-rb")
    parser.add_argument("--model-id", default=sam.DEFAULT_MODEL_ID)
    parser.add_argument("--image", default=str(sam.DEFAULT_IMAGE))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    compile_decoder(args.target_device, args.model_id, args.image, args.device)
