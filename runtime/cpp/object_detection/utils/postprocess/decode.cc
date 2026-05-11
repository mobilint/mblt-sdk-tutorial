// =============================================================================
// decode.cc - YOLO anchorless DET 후처리 구현
// =============================================================================
#include "decode.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float invsigmoid(float x) {
    return -std::log(1.0f / x - 1.0f);
}

YoloDecoder::YoloDecoder(int nc, int nl, int img_size, int reg_max,
                         float conf_thres, float iou_thres, int max_det)
    : nc_(nc), nl_(nl), img_size_(img_size), reg_max_(reg_max),
      conf_thres_(conf_thres), iou_thres_(iou_thres), max_det_(max_det) {
    invconf_ = invsigmoid(conf_thres_);

    // stride: 2^(3+i) -> [8, 16, 32, ...]
    strides_.resize(nl_);
    for (int i = 0; i < nl_; ++i) {
        strides_[i] = 1 << (3 + i);
    }
    // anchor 평면화: 격자 (cx + 0.5, cy + 0.5)
    for (int s : strides_) {
        int gh = img_size_ / s;
        int gw = img_size_ / s;
        grid_sizes_.push_back(gh * gw);
        for (int y = 0; y < gh; ++y) {
            for (int x = 0; x < gw; ++x) {
                anchors_.emplace_back(x + 0.5f, y + 0.5f);
                stride_per_anchor_.push_back(static_cast<float>(s));
            }
        }
    }
}

// 1D NPU raw output 을 stride 별 box / cls 로 분류.
// box: size = reg_max*4 * H*W,  cls: size = nc * H*W.
struct StagedTensor {
    const float* data;
    int channels;
    int hw;
    int stride;
    bool is_cls;
};

static std::vector<StagedTensor> stage_outputs(
    const std::vector<std::vector<float>>& raw,
    int nc, int reg_max, int img_size,
    const std::vector<int>& strides) {
    std::vector<StagedTensor> det_tensors;
    std::vector<StagedTensor> cls_tensors;
    int box_ch = reg_max * 4;
    for (const auto& t : raw) {
        size_t n = t.size();
        for (int s : strides) {
            int gh = img_size / s;
            int gw = img_size / s;
            int hw = gh * gw;
            if (n == static_cast<size_t>(box_ch) * hw) {
                det_tensors.push_back({t.data(), box_ch, hw, s, false});
                break;
            }
            if (n == static_cast<size_t>(nc) * hw) {
                cls_tensors.push_back({t.data(), nc, hw, s, true});
                break;
            }
        }
    }
    // stride 오름차순 (8, 16, 32) 으로 anchor 순서와 일치시킨다.
    auto by_stride = [](const StagedTensor& a, const StagedTensor& b) {
        return a.stride < b.stride;
    };
    std::sort(det_tensors.begin(), det_tensors.end(), by_stride);
    std::sort(cls_tensors.begin(), cls_tensors.end(), by_stride);

    if (det_tensors.size() != cls_tensors.size()) {
        throw std::runtime_error(
            "decode: det/cls tensor count mismatch (det=" +
            std::to_string(det_tensors.size()) + " cls=" +
            std::to_string(cls_tensors.size()) + ")");
    }
    std::vector<StagedTensor> ordered;
    ordered.reserve(det_tensors.size() * 2);
    for (size_t i = 0; i < det_tensors.size(); ++i) {
        ordered.push_back(det_tensors[i]);
        ordered.push_back(cls_tensors[i]);
    }
    return ordered;
}

// IoU (xyxy, xyxy)
static inline float iou_xyxy(float ax1, float ay1, float ax2, float ay2,
                             float bx1, float by1, float bx2, float by2) {
    float ix1 = std::max(ax1, bx1);
    float iy1 = std::max(ay1, by1);
    float ix2 = std::min(ax2, bx2);
    float iy2 = std::min(ay2, by2);
    float iw = std::max(0.0f, ix2 - ix1);
    float ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float ua = std::max(0.0f, ax2 - ax1) * std::max(0.0f, ay2 - ay1);
    float ub = std::max(0.0f, bx2 - bx1) * std::max(0.0f, by2 - by1);
    float denom = ua + ub - inter + 1e-9f;
    return inter / denom;
}

std::vector<YoloDecoder::Detection> YoloDecoder::decode(
    const std::vector<std::vector<float>>& raw_outputs) const {
    const int total_anchors = static_cast<int>(anchors_.size());
    if (total_anchors == 0) return {};

    auto staged = stage_outputs(raw_outputs, nc_, reg_max_, img_size_, strides_);
    if (staged.empty()) return {};

    // anchor 마다 cls logits max 와 box logits 위치 포인터를 모은다.
    // staged 는 [det0, cls0, det1, cls1, ...] 순서.
    struct AnchorAccess {
        const float* box_base;   // (reg_max*4, hw) 의 시작점
        const float* cls_base;   // (nc, hw) 의 시작점
        int hw;
        int local;               // 이 anchor 가 속한 격자 안의 인덱스 (0..hw-1)
    };
    std::vector<AnchorAccess> access(total_anchors);

    int anchor_idx = 0;
    for (size_t st = 0; st < staged.size(); st += 2) {
        const auto& det = staged[st];
        const auto& cls = staged[st + 1];
        for (int i = 0; i < det.hw; ++i) {
            access[anchor_idx] = {det.data, cls.data, det.hw, i};
            ++anchor_idx;
        }
    }

    // 1) cls logit max > invconf 인 anchor 만 선별.
    std::vector<int> active;
    active.reserve(total_anchors);
    for (int a = 0; a < total_anchors; ++a) {
        const float* cls_base = access[a].cls_base;
        int hw = access[a].hw;
        int local = access[a].local;
        float max_logit = cls_base[local];
        for (int c = 1; c < nc_; ++c) {
            float v = cls_base[c * hw + local];
            if (v > max_logit) max_logit = v;
        }
        if (max_logit > invconf_) active.push_back(a);
    }
    if (active.empty()) return {};

    // 2) DFL decode + sigmoid(cls). 결과로 (anchor, cls_pos, cls_neg) detection.
    std::vector<Detection> dets;
    dets.reserve(active.size() * 2);

    std::vector<float> dfl_logits(reg_max_);
    std::vector<float> dfl_softmax(reg_max_);

    for (int a : active) {
        const auto& acc = access[a];
        int hw = acc.hw;
        int local = acc.local;

        // DFL: 4 변 (left, top, right, bottom). reg_max channels each.
        float dist[4];
        for (int side = 0; side < 4; ++side) {
            float maxv = -std::numeric_limits<float>::infinity();
            for (int r = 0; r < reg_max_; ++r) {
                float v = acc.box_base[(side * reg_max_ + r) * hw + local];
                dfl_logits[r] = v;
                if (v > maxv) maxv = v;
            }
            float sum = 0.0f;
            for (int r = 0; r < reg_max_; ++r) {
                dfl_softmax[r] = std::exp(dfl_logits[r] - maxv);
                sum += dfl_softmax[r];
            }
            float exp_dist = 0.0f;
            float inv_sum = 1.0f / sum;
            for (int r = 0; r < reg_max_; ++r) {
                exp_dist += (dfl_softmax[r] * inv_sum) * static_cast<float>(r);
            }
            dist[side] = exp_dist;
        }

        // anchor (cx, cy) + stride scaling
        float cx = anchors_[a].first;
        float cy = anchors_[a].second;
        float st = stride_per_anchor_[a];
        float x1 = (cx - dist[0]) * st;
        float y1 = (cy - dist[1]) * st;
        float x2 = (cx + dist[2]) * st;
        float y2 = (cy + dist[3]) * st;

        // sigmoid(cls) 후 conf_thres 위 클래스 모두 detection 으로 추가.
        for (int c = 0; c < nc_; ++c) {
            float logit = acc.cls_base[c * hw + local];
            if (logit <= invconf_) continue;
            float conf = sigmoid(logit);
            if (conf <= conf_thres_) continue;
            dets.push_back({x1, y1, x2, y2, conf, c});
        }
    }
    if (dets.empty()) return {};

    // 3) 상위 max_pre 개로 컷 (Python 은 30000 제한). 안전한 한도.
    constexpr int max_pre = 30000;
    if (static_cast<int>(dets.size()) > max_pre) {
        std::partial_sort(
            dets.begin(), dets.begin() + max_pre, dets.end(),
            [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
        dets.resize(max_pre);
    } else {
        std::sort(dets.begin(), dets.end(),
                  [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
    }

    // 4) 클래스별 offset 적용 NMS (max_wh = 7680).
    constexpr float max_wh = 7680.0f;
    std::vector<Detection> out;
    out.reserve(std::min<int>(max_det_, static_cast<int>(dets.size())));
    std::vector<char> suppressed(dets.size(), 0);

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        out.push_back(dets[i]);
        if (static_cast<int>(out.size()) >= max_det_) break;
        float ax1 = dets[i].x1 + dets[i].cls * max_wh;
        float ay1 = dets[i].y1 + dets[i].cls * max_wh;
        float ax2 = dets[i].x2 + dets[i].cls * max_wh;
        float ay2 = dets[i].y2 + dets[i].cls * max_wh;
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            float bx1 = dets[j].x1 + dets[j].cls * max_wh;
            float by1 = dets[j].y1 + dets[j].cls * max_wh;
            float bx2 = dets[j].x2 + dets[j].cls * max_wh;
            float by2 = dets[j].y2 + dets[j].cls * max_wh;
            if (iou_xyxy(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) > iou_thres_) {
                suppressed[j] = 1;
            }
        }
    }
    return out;
}

void YoloDecoder::scale_to_original(std::vector<Detection>& dets,
                                    int img_size, int orig_h, int orig_w) {
    float r = std::min(static_cast<float>(img_size) / orig_h,
                       static_cast<float>(img_size) / orig_w);
    int new_h = static_cast<int>(std::round(orig_h * r));
    int new_w = static_cast<int>(std::round(orig_w * r));
    float dh = (img_size - new_h) / 2.0f;
    float dw = (img_size - new_w) / 2.0f;
    for (auto& d : dets) {
        d.x1 = std::clamp((d.x1 - dw) / r, 0.0f, static_cast<float>(orig_w));
        d.x2 = std::clamp((d.x2 - dw) / r, 0.0f, static_cast<float>(orig_w));
        d.y1 = std::clamp((d.y1 - dh) / r, 0.0f, static_cast<float>(orig_h));
        d.y2 = std::clamp((d.y2 - dh) / r, 0.0f, static_cast<float>(orig_h));
    }
}
