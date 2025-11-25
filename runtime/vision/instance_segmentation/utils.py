import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

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
