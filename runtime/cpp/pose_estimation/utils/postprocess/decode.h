// YOLO anchorless pose post-processor: converts raw NPU output tensors into
// (x1, y1, x2, y2, conf, cls) detections plus per-detection keypoints using DFL
// box decode, anchor-based keypoint decode, and class-offset NMS.
// Targets YOLOv8/v11 anchor-free Pose heads only (single "person" class, nc=1).
//
// Algorithm:
//   1) Split raw outputs by stride into box (reg_max*4 channels), cls (nc channels),
//      and kpt (num_keypoints*3 channels) tensors.
//   2) Flatten anchors across all strides -> total_anchors grid points.
//   3) Pre-filter by max cls logit > invsigmoid(conf_thres) to skip most anchors cheaply.
//   4) DFL softmax + expectation -> 4-side distances -> bbox (xyxy, letterbox coordinates).
//   5) sigmoid(cls) then filter conf > conf_thres.
//   6) Decode keypoints: (raw_xy * 2 + anchor - 0.5) * stride for coords, sigmoid for score.
//   7) Apply per-class coordinate offset then run NMS (keypoints follow the surviving box).
//
// (KR) YOLO anchorless 포즈 후처리기: NPU raw 출력 텐서를 DFL 박스 디코드, anchor 기반
// 키포인트 디코드, 클래스별 오프셋 NMS 로 (x1, y1, x2, y2, conf, cls) 탐지 결과와
// 탐지별 키포인트로 변환한다. YOLOv8/v11 anchor-free Pose 헤드 전용 (단일 "person" 클래스, nc=1).
//
// 알고리즘:
//   1) raw 출력을 stride 별 box(reg_max*4 채널), cls(nc 채널), kpt(num_keypoints*3 채널) 텐서로 분리.
//   2) 모든 stride 의 anchor 를 평면화 -> total_anchors 격자 포인트.
//   3) cls 최대 logit > invsigmoid(conf_thres) 로 대부분의 anchor 를 저렴하게 사전 필터링.
//   4) DFL softmax + expectation -> 4변 거리 -> bbox (xyxy, letterbox 좌표).
//   5) sigmoid(cls) 후 conf > conf_thres 필터링.
//   6) 키포인트 디코드: 좌표는 (raw_xy * 2 + anchor - 0.5) * stride, score 는 sigmoid.
//   7) 클래스별 좌표 오프셋 적용 후 NMS 실행 (키포인트는 살아남은 박스를 따라간다).
#pragma once
#include <vector>

class YoloPoseDecoder {
public:
    // One keypoint in letterbox (decode) or original (after scale) coordinates.
    // (KR: letterbox(디코드) 또는 원본(스케일 후) 좌표계의 키포인트 하나.)
    struct Keypoint {
        float x, y, score;
    };

    struct Detection {
        float x1, y1, x2, y2, conf;
        int cls;
        std::vector<Keypoint> kpts;  // num_keypoints entries (KR: num_keypoints 개 항목)
    };

    YoloPoseDecoder(int nc, int nl, int img_size, int reg_max, int num_keypoints,
                    float conf_thres = 0.25f, float iou_thres = 0.7f,
                    int max_det = 300);

    // Decodes raw NPU output tensors (N flat float32 vectors) into detections with
    // keypoints, all in letterbox coordinates.
    // (KR: NPU raw 출력 텐서(평면 float32 벡터 N개)를 키포인트 포함 탐지 결과(letterbox 좌표계)로 디코드한다.)
    std::vector<Detection> decode(
        const std::vector<std::vector<float>>& raw_outputs) const;

    // Rescales detections (boxes and keypoints) from letterbox (img_size x img_size)
    // space to original image coordinates.
    // (KR: 탐지 결과(박스와 키포인트)를 letterbox(img_size x img_size) 좌표에서 원본 이미지 좌표로 변환한다.)
    static void scale_to_original(std::vector<Detection>& dets,
                                  int img_size, int orig_h, int orig_w);

private:
    int nc_;
    int nl_;
    int img_size_;
    int reg_max_;
    int num_keypoints_;
    float conf_thres_;
    float iou_thres_;
    int max_det_;
    float invconf_;                            // invsigmoid(conf_thres) for logit-space pre-filter (KR: logit 공간 사전 필터용)
    std::vector<int> strides_;                 // per-stride values [8, 16, 32, ...] (KR: stride 별 값)
    std::vector<int> grid_sizes_;              // per-stride H*W grid cell counts (KR: stride 별 격자 셀 수)
    std::vector<std::pair<float, float>> anchors_;  // flattened anchor centers (cx, cy) across all strides (KR: 모든 stride 평면화 anchor 중심)
    std::vector<float> stride_per_anchor_;     // stride value for each anchor entry (KR: anchor 별 stride 값)
};
