from argparse import ArgumentParser

from qbcompiler import CalibrationConfig, PreprocessingConfig, Uint8InputConfig, mblt_compile, mxq_compile


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
    parser = ArgumentParser(description="Compile ResNet-50 ONNX model to MXQ model")
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="./resnet50.onnx",
        help="Path to the ONNX model",
    )
    parser.add_argument(
        "--calib-data-path",
        type=str,
        default="./imagenet-1k-selected",
        help="Path to the calibration data",
    )
    parser.add_argument(
        "--mblt-path",
        type=str,
        default="./resnet50.mblt",
        help="Path to save the MBLT model",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./resnet50.mxq",
        help="Path to save the MXQ model",
    )
    parser.add_argument(
        "--target-device",
        type=str,
        choices=["regulus-ra", "regulus-rb", "aries-rb"],
        default="aries-rb",
        help="Target NPU (e.g. aries-rb, regulus-rb)",
    )

    args = parser.parse_args()

    preprocess_pipeline = [
        {"op": "resize", "height": 256, "width": 256, "mode": "bilinear"},
        {"op": "centerCrop", "height": 224, "width": 224},
        {
            "op": "normalize",
            "scaleToUint8": True,  # [0, 255] -> [0, 1]
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "fuseIntoFirstLayer": True,
        },
    ]  # preprocessing operations for resnet 50

    preprocessing_config = PreprocessingConfig(
        apply=True,
        auto_convert_format=True,
        pipeline=preprocess_pipeline,
        input_configs={},
    )

    calibration_config = CalibrationConfig(
        method=1,  # 0 for per tensor, 1 for per channel
        output=0,  # 0 for layer, 1 for channel
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
