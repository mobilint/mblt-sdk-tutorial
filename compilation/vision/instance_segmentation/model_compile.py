from argparse import ArgumentParser

from qubee import QuantizationConfig, mxq_compile

if __name__ == "__main__":
    parser = ArgumentParser(description="Compile YOLO11-seg ONNX model to MXQ model")
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="./yolo11m-seg.onnx",
        help="Path to the ONNX model",
    )
    parser.add_argument(
        "--calib_data_path",
        type=str,
        default="./yolo11m-seg_cali",
        help="Path to the calibration data",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./yolo11m-seg.mxq",
        help="Path to save the MXQ model",
    )
    parser.add_argument(
        "--quant_percentile", type=float, default=0.999, help="Quantization percentile"
    )
    parser.add_argument("--topk_ratio", type=float, default=0.01, help="Top-k ratio")
    parser.add_argument(
        "--inference_scheme",
        type=str,
        choices=["single", "multi", "global", "global4", "global8"],
        default="single",
        help="Inference scheme",
    )

    args = parser.parse_args()

    quantization_config = QuantizationConfig.from_kwargs(
        quantization_method=1,  # 0 for per tensor, 1 for per channel
        quantization_output=1,  # 0 for layer, 1 for channel
        quantization_mode=2,  # maxpercentile
        percentile=args.quant_percentile,
        topk_ratio=args.topk_ratio,
    )

    mxq_compile(
        model=args.onnx_path,
        calib_data_path=args.calib_data_path,
        save_subgraph_type=2,  # save mblt file before quantization
        output_subgraph_path=args.onnx_path.replace(".onnx", ".mblt"),
        save_path=args.save_path,
        backend="onnx",
        inference_scheme=args.inference_scheme,
        quantization_config=quantization_config,
    )
