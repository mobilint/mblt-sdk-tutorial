// Hardcoded ModelInfo for anchorless YOLO face detection (yolov12m-face, P5 head, single "face" class).
// The ultralytics anchor-free Detect head outputs 3 strides x [reg_max*4 box + nc cls] channels;
// for face nc=1, so each stride contributes [64 box + 1 cls] channels.
// Not applicable to P6 variants (num_layers=4).
//
// (KR) anchorless YOLO 얼굴 탐지용 하드코딩 ModelInfo (yolov12m-face, P5 헤드, 단일 "face" 클래스).
// ultralytics anchor-free Detect 헤드는 3 stride x [reg_max*4 box + nc cls] 채널을 출력한다.
// 얼굴은 nc=1 이므로 stride 당 [64 box + 1 cls] 채널이다. P6 변형(num_layers=4)에는 적용 불가.
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
    // 0.25 / 0.45 match the python face tutorial defaults (inference_mxq.py) (KR: python 얼굴 튜토리얼 기본값과 일치)
    cfg.m_postprocess.conf_thres = 0.25f;
    cfg.m_postprocess.iou_thres = 0.45f;
    return cfg;
}
