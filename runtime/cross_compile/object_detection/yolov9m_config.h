// YOLOv9m model configuration for inference tutorial.
//
// YAML 파싱 없이 ModelInfo 를 직접 구성한다.
// inference-regulus/model_configs/det/yolov9m.yaml 과 동일한 값.
#pragma once
#include "parser.h"

static const int IMG_SIZE = 640;

inline ModelInfo make_yolov9m_config() {
    ModelInfo cfg;
    cfg.m_preprocess_list.push_back(
        {PreProcessOps::YOLO, "", std::pair<int, int>{IMG_SIZE, IMG_SIZE}});
    cfg.m_postprocess.task = Task::DET;
    cfg.m_postprocess.type = "yolo";
    cfg.m_postprocess.num_classes = 80;
    cfg.m_postprocess.num_layers = 3;
    cfg.m_postprocess.reg_max = 16;
    // predict_det.cc (mAP 평가) 는 conf=0.001 을 사용하지만,
    // 시각화 튜토리얼에서는 저신뢰 bbox 가 화면을 뒤덮으므로 0.25 로 설정.
    cfg.m_postprocess.conf_thres = 0.25f;
    cfg.m_postprocess.iou_thres = 0.7f;
    return cfg;
}
