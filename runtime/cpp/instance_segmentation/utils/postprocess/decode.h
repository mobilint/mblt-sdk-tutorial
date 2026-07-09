// YOLO anchorless instance-segmentation post-processor: converts raw NPU output tensors into
// (x1, y1, x2, y2, conf, cls) detections plus per-detection mask coefficients, then assembles
// instance masks from the prototype tensor.
// Targets YOLOv8/v11 anchor-free Segment heads only (not anchor-based variants).
//
// Algorithm:
//   1) Split raw outputs by stride into box (reg_max*4 ch), cls (nc ch), and mask-coeff (num_mask_coeffs ch)
//      tensors, plus one prototype tensor [num_mask_coeffs, proto_h, proto_w] (the largest num_mask_coeffs map).
//   2) Flatten anchors across all strides -> total_anchors grid points.
//   3) Pre-filter by max cls logit > invsigmoid(conf_thres) to skip most anchors cheaply.
//   4) DFL softmax + expectation -> 4-side distances -> bbox (xyxy, letterbox coordinates).
//   5) sigmoid(cls) then filter conf > conf_thres; carry the matching mask coefficients.
//   6) Apply per-class coordinate offset then run NMS.
//   7) mask = sigmoid(coeff @ prototypes) -> upsample to img_size -> crop to box -> threshold > 0.0.
//   8) Strip letterbox padding then resize masks to original image and threshold > 0.5.
//
// (KR) YOLO anchorless 인스턴스 분할 후처리기: NPU raw 출력 텐서를 (x1, y1, x2, y2, conf, cls) 탐지
// 결과와 탐지별 마스크 계수로 변환한 뒤, prototype 텐서로 인스턴스 마스크를 조립한다.
// YOLOv8/v11 anchor-free Segment 헤드 전용 (앵커 기반 변형 불가).
//
// 알고리즘:
//   1) raw 출력을 stride 별 box(reg_max*4 채널), cls(nc 채널), mask-coeff(num_mask_coeffs 채널) 텐서와
//      prototype 텐서[num_mask_coeffs, proto_h, proto_w](가장 큰 num_mask_coeffs 맵)로 분리.
//   2) 모든 stride 의 anchor 를 평면화 -> total_anchors 격자 포인트.
//   3) cls 최대 logit > invsigmoid(conf_thres) 로 대부분의 anchor 를 저렴하게 사전 필터링.
//   4) DFL softmax + expectation -> 4변 거리 -> bbox (xyxy, letterbox 좌표).
//   5) sigmoid(cls) 후 conf > conf_thres 필터링; 대응 마스크 계수를 함께 보관.
//   6) 클래스별 좌표 오프셋 적용 후 NMS 실행.
//   7) mask = sigmoid(coeff @ prototypes) -> img_size 로 업샘플 -> box 로 crop -> threshold > 0.0.
//   8) letterbox padding 제거 후 원본 이미지 크기로 resize, threshold > 0.5.
#pragma once
#include <vector>

#include <opencv2/opencv.hpp>

class YoloSegDecoder {
public:
    struct Detection {
        float x1, y1, x2, y2, conf;
        int cls;
        std::vector<float> mask_coeffs;  // num_mask_coeffs mask coefficients (KR: num_mask_coeffs 개 마스크 계수)
    };

    YoloSegDecoder(int nc, int nl, int img_size, int reg_max = 16,
                   int num_mask_coeffs = 32, float conf_thres = 0.25f,
                   float iou_thres = 0.45f, int max_det = 300);

    // Decodes raw NPU output tensors (N flat float32 vectors) into detections in letterbox coordinates.
    // Fills proto_out with the prototype tensor data and sets proto_c/proto_h/proto_w to its dimensions.
    // (KR: NPU raw 출력 텐서를 letterbox 좌표계 탐지 결과로 디코드한다.
    // proto_out 에 prototype 텐서 데이터를 채우고 proto_c/proto_h/proto_w 에 차원을 기록한다.)
    std::vector<Detection> decode(
        const std::vector<std::vector<float>>& raw_outputs,
        std::vector<float>& proto_out,
        int& proto_c, int& proto_h, int& proto_w) const;

    // Assembles one binary instance mask per detection at the original image resolution.
    // Combines mask coefficients with the prototype tensor, upsamples to letterbox space,
    // crops to each box, strips letterbox padding, and resizes to (orig_h, orig_w).
    // (KR: 탐지별 이진 인스턴스 마스크를 원본 해상도로 조립한다. 마스크 계수와 prototype 을 결합해
    // letterbox 공간으로 업샘플, box 로 crop, padding 제거 후 (orig_h, orig_w) 로 resize 한다.)
    // Returns one CV_8U mask (0/255) of size (orig_h, orig_w) per detection.
    // (KR: 탐지마다 크기 (orig_h, orig_w) 의 CV_8U 마스크(0/255) 하나씩 반환.)
    std::vector<cv::Mat> assemble_masks(
        const std::vector<Detection>& dets,
        const std::vector<float>& proto, int proto_c, int proto_h, int proto_w,
        int orig_h, int orig_w) const;

    // Rescales detection boxes from letterbox (img_size x img_size) space to original image coordinates.
    // (KR: 탐지 박스를 letterbox(img_size x img_size) 좌표에서 원본 이미지 좌표로 변환한다.)
    static void scale_to_original(std::vector<Detection>& dets,
                                  int img_size, int orig_h, int orig_w);

private:
    int nc_;
    int nl_;
    int img_size_;
    int reg_max_;
    int num_mask_coeffs_;
    float conf_thres_;
    float iou_thres_;
    int max_det_;
    float invconf_;                            // invsigmoid(conf_thres) for logit-space pre-filter (KR: logit 공간 사전 필터용)
    std::vector<int> strides_;                 // per-stride values [8, 16, 32, ...] (KR: stride 별 값)
    std::vector<int> grid_sizes_;              // per-stride H*W grid cell counts (KR: stride 별 격자 셀 수)
    std::vector<std::pair<float, float>> anchors_;  // flattened anchor centers (cx, cy) across all strides (KR: 모든 stride 평면화 anchor 중심)
    std::vector<float> stride_per_anchor_;     // stride value for each anchor entry (KR: anchor 별 stride 값)
};
