// Hardcoded ModelInfo for anchorless YOLO pose (yolo11m-pose / yolov8m-pose, P5 head, single "person" class).
// The ultralytics v8/v11 anchor-free Pose head shares the same output layout
// (3 strides x [reg_max*4 box + nc cls + num_keypoints*3 kpt] channels), so one config covers both ARIES and REGULUS MXQ files.
// Not applicable to P6 variants (num_layers=4).
//
// (KR) anchorless YOLO 포즈용 하드코딩 ModelInfo (yolo11m-pose / yolov8m-pose, P5 헤드, 단일 "person" 클래스).
// ultralytics v8/v11 의 anchor-free Pose 헤드는 출력 layout 이 동일하므로
// ARIES 와 REGULUS MXQ 파일을 같은 설정으로 처리한다. P6 변형(num_layers=4)에는 적용 불가.
#pragma once
#include "types.h"

static const int IMG_SIZE = 640;

inline ModelInfo make_yolo_pose_config() {
    ModelInfo cfg;
    cfg.m_preprocess_list.push_back(
        {PreProcessOps::YOLO, "", std::pair<int, int>{IMG_SIZE, IMG_SIZE}});
    cfg.m_postprocess.task = Task::POSE;
    cfg.m_postprocess.type = "yolo";
    cfg.m_postprocess.num_classes = 1;       // single "person" class (KR: 단일 "person" 클래스)
    cfg.m_postprocess.num_layers = 3;
    cfg.m_postprocess.reg_max = 16;
    cfg.m_postprocess.num_keypoints = 17;    // COCO 17-keypoint skeleton (KR: COCO 17 키포인트 스켈레톤)
    // 0.25 avoids flooding the visualization with low-confidence poses (KR: 저신뢰 포즈가 화면을 뒤덮는 것을 방지)
    cfg.m_postprocess.conf_thres = 0.25f;
    cfg.m_postprocess.iou_thres = 0.45f;
    return cfg;
}
