import math
import os

import numpy as np
import torch
import torch.nn as nn

NUM_THREADS = min(16, max(1, os.cpu_count() - 1))

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def invsigmoid(x):
    """Inverse sigmoid function."""
    return -math.log(1.0 / x - 1.0)


def make_anchors(imh, imw, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    for stride in strides:
        h, w = imh // stride, imw // stride
        sx = (
            torch.arange(end=w, dtype=torch.float32).to(DEVICE) + grid_cell_offset
        )  # shift x
        sy = (
            torch.arange(end=h, dtype=torch.float32).to(DEVICE) + grid_cell_offset
        )  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(
            torch.full((h * w, 1), stride, dtype=torch.float32).to(DEVICE)
        )
    return torch.cat(anchor_points), torch.cat(stride_tensor)


class DFL(nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data = nn.Parameter(x.view(1, c1, 1, 1)).to(DEVICE)
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape  # batch, channels, anchors
        x = x.to(DEVICE)
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(
            b, 4, a
        )


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def non_max_suppression(boxes, scores, iou_threshold, max_output, eps=1e-9):
    """
    Original source:
    https://github.com/amusi/Non-Maximum-Suppression/blob/master/nms.py

    Modified maximum output to be a parameter, and area/intersection calculation to be correct.

    Performs non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU).

    Args:
        boxes (np.ndarray): The bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        scores (np.ndarray): The confidence scores of the objects, which is sorted in descending order
        iou_threshold (float): The threshold for the IoU
        max_output (int): The maximum number of boxes that will be selected by NMS

    Returns:
        indices (np.ndarray): The indices of the boxes that have been kept by NMS
    """
    if len(boxes) == 0:
        return []

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    picked_indices = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x) * (end_y - start_y)

    order = torch.arange(len(scores)).to(scores.device)

    # Iterate bounding boxes
    while order.numel() > 0 and len(picked_indices) < max_output:
        # The index of largest confidence score
        index = order[0]
        order = order[1:]
        picked_indices.append(index)

        # sanity check
        if len(picked_indices) >= max_output or order.numel() == 0:
            break

        # Compute ordinates of intersection-over-union(IOU)
        x1 = torch.maximum(start_x[index], start_x[order])
        x2 = torch.minimum(end_x[index], end_x[order])
        y1 = torch.maximum(start_y[index], start_y[order])
        y2 = torch.minimum(end_y[index], end_y[order])

        # Compute areas of intersection-over-union
        w = (x2 - x1).clip(0.0)
        h = (y2 - y1).clip(0.0)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order] - intersection + eps)

        order = order[ratio <= iou_threshold]

    return picked_indices


import math
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

_TORCH_VER = [int(x) for x in torch.__version__.split(".")[:2]]
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def make_anchor_grids(nl, imh, imw, anchors):
    """
    Generate data needed for decoding outputs of YOLO with anchors.
    e.g. grids, anchor_grids and strides

    The funcion assumes that the user didn't modify the original
    YOLOv5 models. It believes that anchors used to train model
    are same as those given above. Strides will be implicitly
    deducted from the number of detection layers.

    Args:
        nl: number of detection layers
        imh: input image height
        imw: input image width
        anchors: anchors used to train the model

    Returns:
        all_grids: a list np arrays, coordinates of anchor box centers
        all_anchor_grids: list of np, height and width of anchors boxes
        all_strides: list of ints, ratio between input and output resolutions
    """
    if nl not in [2, 3, 4]:
        raise ValueError(f"Your model has wrong number of detection layers: {nl}")
    na = len(anchors[0]) // 2  # number of anchors

    all_strides = [2 ** (3 + i) for i in range(nl)]
    if nl == 2:  # YOLOv3 tiny has 2 detection layers with strides 32 and 16
        all_strides = [strd * 2 for strd in all_strides]
    out_sizes = [[imh // strd, imw // strd] for strd in all_strides]

    all_grids, all_anchor_grids = [], []
    for anchr, (ny, nx) in zip(anchors, out_sizes):
        y, x = torch.arange(ny, dtype=torch.float32).to(DEVICE), torch.arange(
            nx, dtype=torch.float32
        ).to(DEVICE)
        yv, xv = meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand((1, na, ny, nx, 2))
        all_grids.append(grid)

        anchr = torch.tensor(anchr).view(1, na, 1, 1, 2).to(DEVICE)
        anchr = anchr.expand((1, na, ny, nx, 2))
        all_anchor_grids.append(anchr)

    return all_grids, all_anchor_grids, all_strides


def make_anchors(imh, imw, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    for stride in strides:
        h, w = imh // stride, imw // stride
        sx = (
            torch.arange(end=w, dtype=torch.float32).to(DEVICE) + grid_cell_offset
        )  # shift x
        sy = (
            torch.arange(end=h, dtype=torch.float32).to(DEVICE) + grid_cell_offset
        )  # shift y
        sy, sx = meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(
            torch.full((h * w, 1), stride, dtype=torch.float32).to(DEVICE)
        )
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def meshgrid(*tensors):
    """Generate meshgrid in PyTorch.

    Returns:
        torch.Tensor: meshgrid in PyTorch.
    """
    if _TORCH_VER >= [1, 10]:
        return torch.meshgrid(*tensors, indexing="ij")
    else:
        return torch.meshgrid(*tensors)


class DFL(nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data = nn.Parameter(x.view(1, c1, 1, 1)).to(DEVICE)
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape  # batch, channels, anchors
        x = x.to(DEVICE)
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(
            b, 4, a
        )


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    assert (
        x.shape[-1] == 4
    ), f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = (
        torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
    )  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def masking(nms_outs, proto_outs, img_size):
    imh, imw = img_size
    masks = []
    for pred, proto in zip(nms_outs, proto_outs):
        if len(pred) == 0:
            masks.append(torch.zeros((0, imh, imw), device=DEVICE, dtype=torch.float32))
        else:
            masks.append(
                process_mask_upsample(
                    proto,
                    pred[:, 6:],
                    pred[:, :4],
                    [imh, imw],
                )
            )
    return [[nms_out, mask] for nms_out, mask in zip(nms_outs, masks)]


def process_mask_upsample(protos, masks_in, bboxes, shape):
    """
    Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher quality
    but is slower.

    Args:
        protos (torch.Tensor): [mask_dim, mask_h, mask_w]
        masks_in (torch.Tensor): [n, mask_dim], n is number of masks after nms
        bboxes (torch.Tensor): [n, 4], n is number of masks after nms
        shape (tuple): the size of the input image (h,w)

    Returns:
        (torch.Tensor): The upsampled masks.
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[
        0
    ]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.0).to(torch.float32)


def crop_mask(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, dtype=x1.dtype)[None, None, :].to(masks)  # rows shape(1,w,1)
    c = torch.arange(h, dtype=x1.dtype)[None, :, None].to(masks)  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
