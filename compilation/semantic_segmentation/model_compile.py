"""Compile the YOLO26 semantic-segmentation ONNX model for a Mobilint NPU."""

from argparse import ArgumentParser

from qbcompiler import (
    CalibrationConfig,
    PreprocessingConfig,
    Uint8InputConfig,
    mblt_compile,
    mxq_compile,
)


def get_device_inference_scheme(target_device: str) -> str:
    """Return the inference scheme supported by ``target_device``."""
    if "regulus" in target_device:
        return "single"
    if "aries" in target_device:
        return "all"
    raise ValueError(f"{target_device} is not supported by the current qbcompiler version")


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Compile a YOLO26 semantic-segmentation ONNX model to MXQ and MBLT")
    parser.add_argument("--onnx-path", default="./yolo26m-sem.onnx", help="Path to the ONNX model")
    parser.add_argument(
        "--calib-data-path",
        default="./cityscapes-selected",
        help="Path to the RGB calibration images",
    )
    parser.add_argument("--save-path", default="./yolo26m-sem.mxq", help="Path for the MXQ model")
    parser.add_argument("--mblt-path", default="./yolo26m-sem.mblt", help="Path for the MBLT graph")
    parser.add_argument(
        "--target-device",
        choices=["regulus-ra", "regulus-rb", "aries-rb"],
        default="aries-rb",
        help="Target NPU",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()

    preprocessing_config = PreprocessingConfig(
        apply=True,
        auto_convert_format=True,
        pipeline=[{"op": "letterbox", "height": 1024, "width": 2048, "padValue": 114}],
        input_configs={},
    )
    calibration_config = CalibrationConfig(
        method=1,
        output=1,
        mode=1,
        max_percentile={"percentile": 0.9999, "topk_ratio": 0.01},
    )

    mblt_compile(
        model=args.onnx_path,
        mblt_save_path=args.mblt_path,
        target_device=args.target_device,
        backend="onnx",
        device="cpu",
    )
    mxq_compile(
        model=args.onnx_path,
        calib_data_path=args.calib_data_path,
        save_path=args.save_path,
        image_channels=3,
        backend="onnx",
        device="gpu",
        target_device=args.target_device,
        inference_scheme=get_device_inference_scheme(args.target_device),
        preprocessing_config=preprocessing_config,
        uint8_input_config=Uint8InputConfig(apply=True, inputs=[]),
        calibration_config=calibration_config,
    )


if __name__ == "__main__":
    main()
