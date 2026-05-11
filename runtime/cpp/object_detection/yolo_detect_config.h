// Hardcoded ModelInfo for anchorless YOLO detection (yolo11m / yolov9m, P5 head, 80 COCO classes).
// The ultralytics v8/v9/v11 anchor-free Detect head shares the same output layout
// (3 strides x [reg_max*4 box + nc cls] channels), so one config covers both ARIES and REGULUS MXQ files.
// Not applicable to P6 variants (num_layers=4).
//
// (KR) anchorless YOLO 탐지용 하드코딩 ModelInfo (yolo11m / yolov9m, P5 헤드, COCO 80 클래스).
// ultralytics v8/v9/v11 의 anchor-free Detect 헤드는 출력 layout 이 동일하므로
// ARIES 와 REGULUS MXQ 파일을 같은 설정으로 처리한다. P6 변형(num_layers=4)에는 적용 불가.
#pragma once
#include "types.h"

static const int IMG_SIZE = 640;

inline ModelInfo make_yolo_detect_config() {
    ModelInfo cfg;
    cfg.m_preprocess_list.push_back(
        {PreProcessOps::YOLO, "", std::pair<int, int>{IMG_SIZE, IMG_SIZE}});
    cfg.m_postprocess.task = Task::DET;
    cfg.m_postprocess.type = "yolo";
    cfg.m_postprocess.num_classes = 80;
    cfg.m_postprocess.num_layers = 3;
    cfg.m_postprocess.reg_max = 16;
    // 0.25 avoids flooding the visualization with low-confidence boxes (KR: 저신뢰 박스가 화면을 뒤덮는 것을 방지)
    cfg.m_postprocess.conf_thres = 0.25f;
    cfg.m_postprocess.iou_thres = 0.7f;
    return cfg;
}
