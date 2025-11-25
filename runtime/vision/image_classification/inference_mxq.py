from argparse import ArgumentParser

import maccel
import numpy as np
import torch
import torchvision.transforms.functional as F
from imagenet import get_imagenet_label
from PIL import Image
from torchvision.transforms import InterpolationMode

if __name__ == "__main__":
    parser = ArgumentParser(description="Run inference with compiled model")
    parser.add_argument(
        "--mxq_path", type=str, required=True, help="Path to the compiled MXQ model"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image"
    )
    args = parser.parse_args()

    # ----Load model
    acc = maccel.Accelerator(0)
    mc = maccel.ModelConfig()
    mc.set_single_core_mode(1)
    mxq_model = maccel.Model(args.mxq_path, mc)
    mxq_model.launch(acc)

    # ----Load image
    def preprocess_resnet50(img_path: str):
        img = Image.open(img_path).convert("RGB")
        resize_size = 256
        crop_size = (224, 224)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        out = F.pil_to_tensor(img)
        out = F.resize(out, size=resize_size, interpolation=InterpolationMode.BILINEAR)
        out = F.center_crop(out, output_size=crop_size)
        out = out.to(torch.float, copy=False) / 255.0
        out = F.normalize(out, mean, std)
        out = np.transpose(out.numpy(), axes=[1, 2, 0])
        return out

    image = preprocess_resnet50(args.image_path)
    print("Image shape:", image.shape)

    # ----Run inference
    output = mxq_model.infer(image)

    # ----Print inference results
    print(output[0].shape)
    output = output[0].reshape(-1)

    print("Inference results".center(50, "-"))
    # print the top 5 results
    top5_indices = np.argsort(output)[::-1][:5]
    for idx in top5_indices:
        print(f"{get_imagenet_label(idx)} with probability {output[idx]:.4f}%")
