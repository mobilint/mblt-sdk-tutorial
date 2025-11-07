from argparse import ArgumentParser
import cv2
import numpy as np
from qubee.calibration import make_calib_man

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./val2017")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--save_dir", type=str, default="./")
    parser.add_argument("--save_name", type=str, default="yolo11m-seg_cali")
    parser.add_argument("--max_size", type=int, default=100)

    args = parser.parse_args()

    IMG_SIZE = [args.img_size, args.img_size]  # or [1280, 1280]

    def preprocess_yolo(img_path: str):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h0, w0 = img.shape[:2]  # original hw
        r = min(IMG_SIZE[0] / h0, IMG_SIZE[1] / w0)  # ratio
        new_unpad = int(round(w0 * r)), int(round(h0 * r))
        dh, dw = (
            IMG_SIZE[0] - new_unpad[1],
            IMG_SIZE[1] - new_unpad[0],
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
        pre_ftn=preprocess_yolo,  # callable function to pre-process the calibration data
        data_dir=args.data_dir,  # path to folder of original calibration data files such as images
        save_dir=args.save_dir,  # path to folder to save pre-processed calibration data files
        save_name=args.save_name,  # tag for the generated calibration dataset
        seed=42,  # seed for random selection of calibration data
        max_size=args.max_size,  # Maximum number of data to use for calibration
    )
