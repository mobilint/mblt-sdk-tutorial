// Hardcoded ModelInfo for anchorless YOLO instance segmentation (yolo11m-seg / yolov8m-seg, P5 head, 80 COCO classes).
// The ultralytics v8/v11 anchor-free Segment head extends the Detect head with a mask branch:
// per stride [reg_max*4 box + nc cls + num_mask_coeffs mask] channels, plus one prototype tensor
// of shape [num_mask_coeffs, proto_h, proto_w]. One config covers both ARIES and REGULUS MXQ files.
// Not applicable to P6 variants (num_layers=4).
#pragma once
#include "types.h"

static const int IMG_SIZE = 640;

inline ModelInfo make_yolo_seg_config() {
    ModelInfo cfg;
    cfg.m_preprocess_list.push_back(
        {PreProcessOps::YOLO, "", std::pair<int, int>{IMG_SIZE, IMG_SIZE}});
    cfg.m_postprocess.task = Task::SEG;
    cfg.m_postprocess.type = "yolo";
    cfg.m_postprocess.num_classes = 80;
    cfg.m_postprocess.num_layers = 3;
    cfg.m_postprocess.reg_max = 16;
    cfg.m_postprocess.num_mask_coeffs = 32;
    // 0.25 avoids flooding the visualization with low-confidence masks
    cfg.m_postprocess.conf_thres = 0.25f;
    cfg.m_postprocess.iou_thres = 0.45f;
    return cfg;
}
