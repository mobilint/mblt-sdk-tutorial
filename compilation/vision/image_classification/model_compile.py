from argparse import ArgumentParser

from qbcompiler import (
    InputProcessConfig,
    PreprocessingConfig,
    QuantizationConfig,
    Uint8InputConfig,
    mxq_compile,
)

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
        "--save-path",
        type=str,
        default="./resnet50.mxq",
        help="Path to save the MXQ model",
    )

    args = parser.parse_args()

    preprocess_pipeline = [
        {"op": "resize", "height": 256, "width": 256, "mode": "bilinear"},
        {"op": "centerCrop", "height": 224, "width": 224},
        {
            "op": "normalize",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "scaleToUint8": True,  # [0, 255] -> [0, 1]
            "fuseIntoFirstLayer": True,
        },
    ]  # preprocessing operations for resnet 50

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
        quantization_output=0,  # 0 for layer, 1 for channel
        quantization_mode=2,  # maxpercentile
        percentile=0.9999,  # quantization percentile
        topk_ratio=0.01,  # quantization topk
    )

    mxq_compile(
        model=args.onnx_path,
        calib_data_path=args.calib_data_path,
        save_path=args.save_path,
        save_subgraph_type=2,  # save mblt file before quantization
        output_subgraph_path=args.onnx_path.replace(".onnx", ".mblt"),
        backend="onnx",
        device="gpu",
        inference_scheme="single",  # single core mode
        input_process_config=input_process_config,
        quantization_config=quantization_config,
    )
