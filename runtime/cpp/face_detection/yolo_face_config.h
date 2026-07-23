// Hardcoded ModelInfo for anchorless YOLO face detection (yolov12m-face, P5 head, single "face" class).
// The ultralytics anchor-free Detect head outputs 3 strides x [reg_max*4 box + nc cls] channels;
// for face nc=1, so each stride contributes [64 box + 1 cls] channels.
// Not applicable to P6 variants (num_layers=4).
#pragma once
#include "types.h"

static const int IMG_SIZE = 640;

inline ModelInfo make_yolo_face_config() {
    ModelInfo cfg;
    cfg.m_preprocess_list.push_back(
        {PreProcessOps::YOLO, "", std::pair<int, int>{IMG_SIZE, IMG_SIZE}});
    cfg.m_postprocess.task = Task::DET;
    cfg.m_postprocess.type = "yolo";
    cfg.m_postprocess.num_classes = 1;
    cfg.m_postprocess.num_layers = 3;
    cfg.m_postprocess.reg_max = 16;
    // 0.25 / 0.45 match the python face tutorial defaults (inference_mxq.py)
    cfg.m_postprocess.conf_thres = 0.25f;
    cfg.m_postprocess.iou_thres = 0.45f;
    return cfg;
}
