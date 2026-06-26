// Hardcoded ModelInfo for anchorless YOLO instance segmentation (yolo11m-seg / yolov8m-seg, P5 head, 80 COCO classes).
// The ultralytics v8/v11 anchor-free Segment head extends the Detect head with a mask branch:
// per stride [reg_max*4 box + nc cls + num_mask_coeffs mask] channels, plus one prototype tensor
// of shape [num_mask_coeffs, proto_h, proto_w]. One config covers both ARIES and REGULUS MXQ files.
// Not applicable to P6 variants (num_layers=4).
//
// (KR) anchorless YOLO 인스턴스 분할용 하드코딩 ModelInfo (yolo11m-seg / yolov8m-seg, P5 헤드, COCO 80 클래스).
// ultralytics v8/v11 의 anchor-free Segment 헤드는 Detect 헤드에 마스크 분기를 더한 것으로,
// stride 별 [reg_max*4 box + nc cls + num_mask_coeffs mask] 채널과 더불어
// shape [num_mask_coeffs, proto_h, proto_w] 의 prototype 텐서를 하나 출력한다.
// ARIES 와 REGULUS MXQ 파일을 같은 설정으로 처리한다. P6 변형(num_layers=4)에는 적용 불가.
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
    // 0.25 avoids flooding the visualization with low-confidence masks (KR: 저신뢰 마스크가 화면을 뒤덮는 것을 방지)
    cfg.m_postprocess.conf_thres = 0.25f;
    cfg.m_postprocess.iou_thres = 0.45f;
    return cfg;
}
