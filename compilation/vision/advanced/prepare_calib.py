import os
from argparse import ArgumentParser

from qubee.calibration import make_calib

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    if args.data_dir.endswith("/"):
        args.data_dir = args.data_dir[:-1]

    make_calib(
        args_pre="./resnet50.yaml",  # path to pre-processing configuration yaml file
        data_dir=args.data_dir,  # path to folder of original calibration data files such as images
        save_dir="./",  # path to folder to save pre-processed calibration data files
        save_name=f"resnet50_{os.path.basename(args.data_dir)}",  # tag for the generated calibration dataset
        max_size=-1,  # Maximum number of data to use for calibration. -1 means all data will be used.
    )
