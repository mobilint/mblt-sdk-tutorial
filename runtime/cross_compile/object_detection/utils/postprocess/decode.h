// =============================================================================
// decode.h - YOLO anchorless DET 후처리 (DFL decode + NMS)
// =============================================================================
// NPU raw output 텐서들을 받아 (x1, y1, x2, y2, conf, cls) detection 으로
// 변환한다. anchorless (YOLOv8/v9, v3u/v5u) 전용.
//
// 알고리즘:
//   1) raw output 을 stride 별 (box reg_max*4 채널, cls nc 채널) 로 분리
//   2) 모든 stride 의 anchor 평면화 -> total_anchors
//   3) cls 최대값으로 conf 사전 필터 (logit > invsigmoid(conf_thres))
//   4) DFL softmax + expectation -> 4변 거리 -> bbox (xyxy, letterbox 좌표)
//   5) sigmoid(cls) 후 conf > thres 필터
//   6) 클래스별 offset 적용 후 NMS
//
// Anchor / DFL / NMS 는 utils/postprocess/decode.py 의 PyTorch 구현과 동등.
// =============================================================================
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

    // raw_outputs: NPU 출력 텐서 N 개 (각각 평면 float32 벡터)
    // 반환: letterbox 좌표계의 detections.
    std::vector<Detection> decode(
        const std::vector<std::vector<float>>& raw_outputs) const;

    // letterbox(img_size x img_size) -> 원본 이미지 좌표계 변환.
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
    float invconf_;                            // invsigmoid(conf_thres)
    std::vector<int> strides_;                 // stride 별 [8,16,32,...]
    std::vector<int> grid_sizes_;              // stride 별 H*W
    std::vector<std::pair<float, float>> anchors_;  // 모든 anchor (cx, cy)
    std::vector<float> stride_per_anchor_;     // anchor 마다 해당 stride
};
