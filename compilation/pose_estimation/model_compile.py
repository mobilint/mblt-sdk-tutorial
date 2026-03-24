from argparse import ArgumentParser

from qbcompiler import (
    CalibrationConfig,
    PreprocessingConfig,
    Uint8InputConfig,
    mxq_compile,
)

if __name__ == "__main__":
    parser = ArgumentParser(description="Compile YOLO11 ONNX model to MXQ model")
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="./yolo11m-pose.onnx",
        help="Path to the ONNX model",
    )
    parser.add_argument(
        "--calib-data-path",
        type=str,
        default="./coco-selected",
        help="Path to the calibration data",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./yolo11m-pose.mxq",
        help="Path to save the MXQ model",
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
    mxq_compile(
        model=args.onnx_path,
        calib_data_path=args.calib_data_path,
        save_subgraph_type=2,  # save mblt file before quantization
        output_subgraph_path=args.onnx_path.replace(".onnx", ".mblt"),
        save_path=args.save_path,
        backend="onnx",
        inference_scheme="single",
        preprocessing_config=preprocessing_config,
        uint8_input_config=Uint8InputConfig(apply=True, inputs=[]),
    )
