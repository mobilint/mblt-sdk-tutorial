from argparse import ArgumentParser

from qbcompiler import (
    CalibrationConfig,
    PreprocessingConfig,
    Uint8InputConfig,
    mblt_compile,
    mxq_compile,
)


def get_device_inference_sheme(target_device):
    # regulus device only support single
    if "regulus" in target_device:
        return "single"
    # aries device support all
    elif "aries" in target_device:
        return "all"
    else:
        raise ValueError(f"{target_device} not supported current qbcompiler version")


if __name__ == "__main__":
    parser = ArgumentParser(description="Compile YOLO Face ONNX model to MXQ / MBLT")
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="./yolov12m-face.onnx",
        help="Path to the ONNX model",
    )
    parser.add_argument(
        "--calib-data-path",
        type=str,
        default="./widerface-selected",
        help="Path to the calibration data",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./yolov12m-face.mxq",
        help="Path to save the MXQ model",
    )
    parser.add_argument(
        "--mblt-path",
        type=str,
        default="./yolov12m-face.mblt",
        help="Path to save the MBLT model",
    )
    parser.add_argument(
        "--target-device",
        type=str,
        choices=["regulus-ra", "regulus-rb", "aries-rb"],
        default="aries-rb",
        help="Target NPU (e.g. aries-rb, regulus-rb)",
    )

    args = parser.parse_args()

    preprocess_pipeline = [{"op": "letterbox", "height": 640, "width": 640, "padValue": 114}]

    preprocessing_config = PreprocessingConfig(
        apply=True,
        auto_convert_format=True,
        pipeline=preprocess_pipeline,
        input_configs={},
    )

    calibration_config = CalibrationConfig(
        method=1,  # 0 for per tensor, 1 for per channel
        output=1,  # 0 for layer, 1 for channel
        mode=1,  # maxpercentile
        max_percentile={
            "percentile": 0.9999,  # quantization percentile
            "topk_ratio": 0.01,  # quantization topk
        },
    )

    # inference_sheme is difference device by device
    inferece_sheme = get_device_inference_sheme(args.target_device)

    # ONNX -> MBLT : intermediate graph only (no quantization), for inspection/visualization
    mblt_compile(
        model=args.onnx_path,
        mblt_save_path=args.mblt_path,
        target_device=args.target_device,
        backend="onnx",
        device="cpu",
    )

    # ONNX -> MXQ : quantized package that runs on the NPU
    mxq_compile(
        model=args.onnx_path,
        calib_data_path=args.calib_data_path,
        save_path=args.save_path,
        image_channels=3,  # If there is grayscale image in calibration dataset, convert to RGB
        backend="onnx",
        device="gpu",
        target_device=args.target_device,
        inference_scheme=inferece_sheme,
        preprocessing_config=preprocessing_config,
        uint8_input_config=Uint8InputConfig(apply=True, inputs=[]),
        calibration_config=calibration_config,
    )
