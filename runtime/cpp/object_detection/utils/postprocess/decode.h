// YOLO anchorless detection post-processor: converts raw NPU output tensors into
// (x1, y1, x2, y2, conf, cls) detections using DFL decode and class-offset NMS.
// Targets YOLOv8/v9/v11 anchor-free Detect heads only (not anchor-based variants).
//
// Algorithm:
//   1) Split raw outputs by stride into box (reg_max*4 channels) and cls (nc channels) tensors.
//   2) Flatten anchors across all strides -> total_anchors grid points.
//   3) Pre-filter by max cls logit > invsigmoid(conf_thres) to skip most anchors cheaply.
//   4) DFL softmax + expectation -> 4-side distances -> bbox (xyxy, letterbox coordinates).
//   5) sigmoid(cls) then filter conf > conf_thres.
//   6) Apply per-class coordinate offset then run NMS.
//
// (KR) YOLO anchorless 탐지 후처리기: NPU raw 출력 텐서를 DFL 디코드와 클래스별 오프셋 NMS 로
// (x1, y1, x2, y2, conf, cls) 탐지 결과로 변환한다.
// YOLOv8/v9/v11 anchor-free Detect 헤드 전용 (앵커 기반 변형 불가).
//
// 알고리즘:
//   1) raw 출력을 stride 별 box(reg_max*4 채널)와 cls(nc 채널) 텐서로 분리.
//   2) 모든 stride 의 anchor 를 평면화 -> total_anchors 격자 포인트.
//   3) cls 최대 logit > invsigmoid(conf_thres) 로 대부분의 anchor 를 저렴하게 사전 필터링.
//   4) DFL softmax + expectation -> 4변 거리 -> bbox (xyxy, letterbox 좌표).
//   5) sigmoid(cls) 후 conf > conf_thres 필터링.
//   6) 클래스별 좌표 오프셋 적용 후 NMS 실행.
#pragma once
#include <vector>

class YoloDecoder {
public:
    struct Detection {
        float x1, y1, x2, y2, conf;
        int cls;
    };

    YoloDecoder(int nc, int nl, int img_size, int reg_max = 16,
                float conf_thres = 0.25f, float iou_thres = 0.7f,
                int max_det = 300);

    // Decodes raw NPU output tensors (N flat float32 vectors) into detections in letterbox coordinates.
    // (KR: NPU raw 출력 텐서(평면 float32 벡터 N개)를 letterbox 좌표계의 탐지 결과로 디코드한다.)
    std::vector<Detection> decode(
        const std::vector<std::vector<float>>& raw_outputs) const;

    // Rescales detections from letterbox (img_size x img_size) space to original image coordinates.
    // (KR: letterbox(img_size x img_size) 좌표를 원본 이미지 좌표계로 변환한다.)
    static void scale_to_original(std::vector<Detection>& dets,
                                  int img_size, int orig_h, int orig_w);

private:
    int nc_;
    int nl_;
    int img_size_;
    int reg_max_;
    float conf_thres_;
    float iou_thres_;
    int max_det_;
    float invconf_;                            // invsigmoid(conf_thres) for logit-space pre-filter (KR: logit 공간 사전 필터용)
    std::vector<int> strides_;                 // per-stride values [8, 16, 32, ...] (KR: stride 별 값)
    std::vector<int> grid_sizes_;              // per-stride H*W grid cell counts (KR: stride 별 격자 셀 수)
    std::vector<std::pair<float, float>> anchors_;  // flattened anchor centers (cx, cy) across all strides (KR: 모든 stride 평면화 anchor 중심)
    std::vector<float> stride_per_anchor_;     // stride value for each anchor entry (KR: anchor 별 stride 값)
};
