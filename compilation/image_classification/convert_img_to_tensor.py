import os
from argparse import ArgumentParser
from typing import cast

import torch
import torchvision.transforms as T
from PIL import Image
from qbcompiler.calibration import make_calib_man

if __name__ == "__main__":
    parser = ArgumentParser(description="Convert images to tensor for calibration")
    parser.add_argument(
        "--source-path",
        type=str,
        default="./imagenet-1k-selected",
        help="Path to the calibration data",
    )
    parser.add_argument(
        "--npy-path",
        type=str,
        default="./calib_data_tensor",
        help="Path to save the tensor data",
    )
    args = parser.parse_args()

    def pre_ftn(img_path):
        img = Image.open(img_path).convert("RGB")
        preprocess_pipeline = [
            T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop((224, 224)),
            T.ToTensor(),  # [0, 255] -> [0, 1]
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        preprocess = T.Compose(preprocess_pipeline)
        tensor = cast(torch.Tensor, preprocess(img))
        return tensor.permute((1, 2, 0)).numpy()  # (C,H,W) -> (H,W,C)

    make_calib_man(
        pre_ftn=pre_ftn,
        data_dir=args.source_path,
        save_dir=os.path.dirname(args.npy_path),
        save_name=os.path.basename(args.npy_path),
        remove_npy=True,  # clean up destination folder before saving new npy files
    )
