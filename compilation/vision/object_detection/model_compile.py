from argparse import ArgumentParser

from qbcompiler import (
    InputProcessConfig,
    PreprocessingConfig,
    QuantizationConfig,
    Uint8InputConfig,
    mxq_compile,
)

if __name__ == "__main__":
    parser = ArgumentParser(description="Compile YOLO11 ONNX model to MXQ model")
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="./yolo11m.onnx",
        help="Path to the ONNX model",
    )
    parser.add_argument(
        "--calib_data_path",
        type=str,
        default="./coco-selected",
        help="Path to the calibration data",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./yolo11m.mxq",
        help="Path to save the MXQ model",
    )

    args = parser.parse_args()

    preprocess_pipeline = [
        {"op": "letterbox", "height": 640, "width": 640, "padValue": 114}
    ]

    preprocessing_config = PreprocessingConfig(
        apply=True,
        auto_convert_format=True,
        pipeline=preprocess_pipeline,
        input_configs={},
    )

    input_process_config = InputProcessConfig(
        uint8_input=Uint8InputConfig(apply=True, inputs=[]),
        image_channels=3,
        preprocessing=preprocessing_config,
    )

    quantization_config = QuantizationConfig.from_kwargs(
        quantization_method=1,  # 0 for per tensor, 1 for per channel
        quantization_output=1,  # 0 for layer, 1 for channel
        quantization_mode=2,  # maxpercentile
        percentile=0.999,
        topk_ratio=0.01,
    )

    mxq_compile(
        model=args.onnx_path,
        calib_data_path=args.calib_data_path,
        save_subgraph_type=2,  # save mblt file before quantization
        output_subgraph_path=args.onnx_path.replace(".onnx", ".mblt"),
        save_path=args.save_path,
        backend="onnx",
        inference_scheme="single",
        input_process_config=input_process_config,
        quantization_config=quantization_config,
    )
