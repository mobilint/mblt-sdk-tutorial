from qubee.calibration import make_calib

make_calib(
    args_pre="./resnet50.yaml",  # path to pre-processing configuration yaml file
    data_dir="./imagenet-1k-selected",  # path to folder of original calibration data files such as images
    save_dir="./",  # path to folder to save pre-processed calibration data files
    save_name="resnet50_cali",  # tag for the generated calibration dataset
    max_size=100,  # Maximum number of data to use for calibration. -1 means all data will be used.
)
