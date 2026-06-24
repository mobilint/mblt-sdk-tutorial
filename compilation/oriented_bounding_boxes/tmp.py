from qbcompiler import mblt_compile

if __name__ == "__main__":
    mblt_compile(
        model="/models/ONNX/obb/yolo26m-obb.onnx",
        target_device="aries-rb",
        mblt_save_path="./yolo26m-obb.mblt",
        yolo_decode_include=True,
    )
