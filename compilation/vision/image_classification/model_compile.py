from argparse import ArgumentParser

from qubee import mxq_compile

if __name__ == "__main__":
    parser = ArgumentParser(description="Compile ResNet-50 ONNX model to MXQ model")
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="./resnet50.onnx",
        help="Path to the ONNX model",
    )
    parser.add_argument(
        "--calib_data_path",
        type=str,
        default="./resnet50_cali",
        help="Path to the calibration data",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./resnet50.mxq",
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
    mxq_compile(
        model=args.onnx_path,
        calib_data_path=args.calib_data_path,
        quantize_method="maxpercentile",  # quantization method to use
        is_quant_ch=True,  # whether to use channel-wise quantization
        quantize_percentile=args.quant_percentile,
        topk_ratio=args.topk_ratio,
        quant_output="layer",  # quantization method for the output layer
        save_path=args.save_path,
        backend="onnx",
        inference_scheme=args.inference_scheme,
        optimize_option=2,
    )
