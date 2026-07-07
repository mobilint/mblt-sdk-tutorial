import os
from argparse import ArgumentParser

import cv2
import numpy as np
from qbcompiler.calibration import make_calib_man

if __name__ == "__main__":
    parser = ArgumentParser(description="Convert images to tensor for calibration")
    parser.add_argument(
        "--source-path",
        type=str,
        default="./coco-selected",
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
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h0, w0 = img.shape[:2]  # original hw
        r = min(640 / h0, 640 / w0)  # ratio
        new_unpad = int(round(w0 * r)), int(round(h0 * r))
        dh, dw = (
            640 - new_unpad[1],
            640 - new_unpad[0],
        )  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if (img.shape[1], img.shape[0]) != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        img = (img / 255).astype(np.float32)

        return img

    make_calib_man(
        pre_ftn=pre_ftn,
        data_dir=args.source_path,
        save_dir=os.path.dirname(args.npy_path),
        save_name=os.path.basename(args.npy_path),
        remove_npy=True,  # clean up destination folder before saving new npy files
    )
