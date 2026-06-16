// Implementation of YoloSegDecoder declared in decode.h.
// Performs DFL box decoding, sigmoid class scoring, class-offset greedy NMS, and instance-mask
// assembly (mask coefficients x prototype tensor -> upsample -> crop -> threshold -> rescale).
//
// (KR) decode.h 에 선언된 YoloSegDecoder 구현.
// NPU 출력에 대해 DFL 박스 디코딩, sigmoid 클래스 스코어링, 클래스별 오프셋 greedy NMS 를 수행하고,
// 인스턴스 마스크를 조립한다(마스크 계수 x prototype 텐서 -> 업샘플 -> crop -> threshold -> rescale).
#include "decode.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float invsigmoid(float x) {
    return -std::log(1.0f / x - 1.0f);
}

YoloSegDecoder::YoloSegDecoder(int nc, int nl, int img_size, int reg_max,
                               int num_mask_coeffs, float conf_thres,
                               float iou_thres, int max_det)
    : nc_(nc), nl_(nl), img_size_(img_size), reg_max_(reg_max),
      num_mask_coeffs_(num_mask_coeffs), conf_thres_(conf_thres),
      iou_thres_(iou_thres), max_det_(max_det) {
    invconf_ = invsigmoid(conf_thres_);

    // Strides follow 2^(3+i): [8, 16, 32, ...] for P3/P4/P5 feature maps.
    // (KR: stride 는 2^(3+i) 패턴: P3/P4/P5 특징맵에 대해 [8, 16, 32, ...].)
    strides_.resize(nl_);
    for (int i = 0; i < nl_; ++i) {
        strides_[i] = 1 << (3 + i);
    }
    // Anchor centers are offset by 0.5 to place them at grid cell centers (KR: anchor 중심을 격자 셀 중앙에 맞추기 위해 0.5 오프셋 적용)
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

// Classifies flat NPU output tensors by stride into box (reg_max*4 ch), cls (nc ch),
// and mask-coeff (num_mask_coeffs ch) groups. The prototype tensor (num_mask_coeffs ch but
// the largest spatial map) is detected separately by the caller.
// (KR: NPU raw 출력 텐서를 stride 별 box(reg_max*4 채널), cls(nc 채널), mask-coeff(num_mask_coeffs 채널)
// 그룹으로 분류한다. prototype 텐서(num_mask_coeffs 채널이지만 가장 큰 공간 맵)는 호출부에서 별도 처리한다.)
struct StagedTensor {
    const float* data;
    int channels;
    int hw;
    int stride;
};

std::vector<YoloSegDecoder::Detection> YoloSegDecoder::decode(
    const std::vector<std::vector<float>>& raw_outputs,
    std::vector<float>& proto_out,
    int& proto_c, int& proto_h, int& proto_w) const {
    const int total_anchors = static_cast<int>(anchors_.size());
    if (total_anchors == 0) return {};

    int box_ch = reg_max_ * 4;

    // Stage tensors by stride. Box and cls tensors map uniquely to a stride by their channel count
    // and H*W. Mask-coeff tensors (num_mask_coeffs ch) come in two kinds: per-stride coeff maps
    // (H*W matches a stride grid) and the single prototype map (largest H*W). The prototype is the
    // num_mask_coeffs-channel tensor whose element count exceeds every per-stride grid.
    // (KR: stride 별로 텐서를 분류. box 와 cls 텐서는 채널 수와 H*W 로 stride 에 유일하게 매핑된다.
    // mask-coeff 텐서(num_mask_coeffs 채널)는 두 종류: stride 별 coeff 맵(H*W 가 stride 격자와 일치)과
    // 단일 prototype 맵(가장 큰 H*W). prototype 은 모든 stride 격자보다 원소 수가 큰 num_mask_coeffs 채널 텐서다.)
    std::vector<StagedTensor> det_tensors;
    std::vector<StagedTensor> cls_tensors;
    std::vector<StagedTensor> ext_tensors;
    const std::vector<float>* proto_src = nullptr;
    int proto_numel = -1;

    for (const auto& t : raw_outputs) {
        size_t n = t.size();
        bool matched = false;
        for (int s : strides_) {
            int gh = img_size_ / s;
            int gw = img_size_ / s;
            int hw = gh * gw;
            if (n == static_cast<size_t>(box_ch) * hw) {
                det_tensors.push_back({t.data(), box_ch, hw, s});
                matched = true;
                break;
            }
            if (n == static_cast<size_t>(nc_) * hw) {
                cls_tensors.push_back({t.data(), nc_, hw, s});
                matched = true;
                break;
            }
            if (n == static_cast<size_t>(num_mask_coeffs_) * hw) {
                ext_tensors.push_back({t.data(), num_mask_coeffs_, hw, s});
                matched = true;
                break;
            }
        }
        if (matched) continue;
        // Unmatched num_mask_coeffs-channel tensor with the largest element count -> prototype.
        // (KR: stride 격자와 안 맞는 num_mask_coeffs 채널 텐서 중 원소 수가 가장 큰 것 -> prototype.)
        if (num_mask_coeffs_ > 0 && n % static_cast<size_t>(num_mask_coeffs_) == 0 &&
            static_cast<int>(n) > proto_numel) {
            proto_src = &t;
            proto_numel = static_cast<int>(n);
        }
    }

    // Sort ascending by stride (8, 16, 32) to match the anchor flattening order in the constructor.
    // (KR: 생성자에서 anchor 를 평면화한 순서와 맞추기 위해 stride 오름차순(8, 16, 32)으로 정렬.)
    auto by_stride = [](const StagedTensor& a, const StagedTensor& b) {
        return a.stride < b.stride;
    };
    std::sort(det_tensors.begin(), det_tensors.end(), by_stride);
    std::sort(cls_tensors.begin(), cls_tensors.end(), by_stride);
    std::sort(ext_tensors.begin(), ext_tensors.end(), by_stride);

    if (det_tensors.size() != cls_tensors.size() ||
        det_tensors.size() != ext_tensors.size()) {
        throw std::runtime_error(
            "decode: det/cls/ext tensor count mismatch (det=" +
            std::to_string(det_tensors.size()) + " cls=" +
            std::to_string(cls_tensors.size()) + " ext=" +
            std::to_string(ext_tensors.size()) + ")");
    }
    if (proto_src == nullptr) {
        throw std::runtime_error("decode: prototype tensor not found");
    }

    // Export the prototype tensor. The seg prototype is [num_mask_coeffs, proto_h, proto_w] in CHW,
    // and proto_h == proto_w == img_size / 4 for the P5 head (e.g. 160x160 at img_size 640).
    // (KR: prototype 텐서 export. seg prototype 은 CHW [num_mask_coeffs, proto_h, proto_w] 이고
    // P5 헤드에서 proto_h == proto_w == img_size / 4 다(예: img_size 640 에서 160x160).)
    proto_c = num_mask_coeffs_;
    int proto_hw = proto_numel / num_mask_coeffs_;
    int proto_side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(proto_hw))));
    proto_h = proto_side;
    proto_w = proto_side;
    if (proto_h * proto_w != proto_hw) {
        throw std::runtime_error("decode: non-square prototype map is not supported");
    }
    proto_out = *proto_src;

    // Build per-anchor access structs aligned with the constructor's stride order.
    // (KR: 생성자의 stride 순서에 맞춘 anchor 별 접근 구조체 구성.)
    struct AnchorAccess {
        const float* box_base;   // (reg_max*4, hw) 의 시작점
        const float* cls_base;   // (nc, hw) 의 시작점
        const float* ext_base;   // (num_mask_coeffs, hw) 의 시작점
        int hw;
        int local;               // index within this stride's grid (0..hw-1) (KR: 이 stride 격자 안의 인덱스)
    };
    std::vector<AnchorAccess> access(total_anchors);

    int anchor_idx = 0;
    for (size_t st = 0; st < det_tensors.size(); ++st) {
        const auto& det = det_tensors[st];
        const auto& cls = cls_tensors[st];
        const auto& ext = ext_tensors[st];
        for (int i = 0; i < det.hw; ++i) {
            access[anchor_idx] = {det.data, cls.data, ext.data, det.hw, i};
            ++anchor_idx;
        }
    }

    // Invariant: every anchor slot in `access` must be populated before the pre-filter loop
    // dereferences it. A partial input (one stride triple missing) would otherwise leave trailing
    // anchor slots uninitialized.
    // (KR: pre-filter 가 access 를 deref 하기 전 모든 anchor slot 이 채워졌는지 확인.
    // 한 stride 묶음이 빠진 입력에서는 뒤쪽 anchor slot 이 초기화되지 않은 상태로 남는다.)
    if (anchor_idx != total_anchors) {
        throw std::runtime_error(
            "YoloSegDecoder::decode: NPU outputs do not cover all anchors ("
            + std::to_string(anchor_idx) + " of "
            + std::to_string(total_anchors) + " populated)");
    }

    // Pre-filter: keep only anchors whose max cls logit exceeds invconf_ (cheap logit-space threshold).
    // (KR: 사전 필터: max cls logit 이 invconf_ 를 초과하는 anchor 만 유지(저렴한 logit 공간 임계값).)
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

    // DFL decode + sigmoid(cls): convert passing anchors to detections, carrying mask coefficients.
    // (KR: DFL 디코드 + sigmoid(cls): 통과 anchor 를 탐지 결과로 변환하고 마스크 계수를 함께 보관.)
    std::vector<Detection> dets;
    dets.reserve(active.size() * 2);

    std::vector<float> dfl_logits(reg_max_);
    std::vector<float> dfl_softmax(reg_max_);

    for (int a : active) {
        const auto& acc = access[a];
        int hw = acc.hw;
        int local = acc.local;

        // DFL softmax over reg_max bins for each of 4 sides (left, top, right, bottom).
        // (KR: 4변(left, top, right, bottom) 각각에 대해 reg_max bin 으로 DFL softmax 적용.)
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

        // Convert DFL distances to xyxy pixel coords: (anchor - left/top) * stride and (anchor + right/bottom) * stride.
        // (KR: DFL 거리를 xyxy 픽셀 좌표로 변환: (anchor - left/top) * stride, (anchor + right/bottom) * stride.)
        float cx = anchors_[a].first;
        float cy = anchors_[a].second;
        float st = stride_per_anchor_[a];
        float x1 = (cx - dist[0]) * st;
        float y1 = (cy - dist[1]) * st;
        float x2 = (cx + dist[2]) * st;
        float y2 = (cy + dist[3]) * st;

        // Emit one Detection per class whose sigmoid score exceeds conf_thres.
        // (KR: sigmoid 점수가 conf_thres 를 초과하는 클래스마다 Detection 을 생성.)
        for (int c = 0; c < nc_; ++c) {
            float logit = acc.cls_base[c * hw + local];
            if (logit <= invconf_) continue;
            float conf = sigmoid(logit);
            if (conf <= conf_thres_) continue;
            Detection d;
            d.x1 = x1;
            d.y1 = y1;
            d.x2 = x2;
            d.y2 = y2;
            d.conf = conf;
            d.cls = c;
            d.mask_coeffs.resize(num_mask_coeffs_);
            for (int k = 0; k < num_mask_coeffs_; ++k) {
                d.mask_coeffs[k] = acc.ext_base[k * hw + local];
            }
            dets.push_back(std::move(d));
        }
    }
    if (dets.empty()) return {};

    // Cap candidates at 30000 before NMS to bound worst-case O(n^2) cost (matches ultralytics default).
    // (KR: NMS 전 후보를 30000 개로 제한해 최악의 O(n^2) 비용을 억제(ultralytics 기본값과 동일).)
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

    // Greedy NMS with per-class coordinate offset (max_wh=7680) so boxes of different classes never suppress each other.
    // (KR: 클래스별 좌표 오프셋(max_wh=7680) 적용 greedy NMS; 다른 클래스 박스끼리는 억제되지 않는다.)
    constexpr float max_wh = 7680.0f;
    auto iou_xyxy = [](float ax1, float ay1, float ax2, float ay2,
                       float bx1, float by1, float bx2, float by2) -> float {
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
    };

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

std::vector<cv::Mat> YoloSegDecoder::assemble_masks(
    const std::vector<Detection>& dets,
    const std::vector<float>& proto, int proto_c, int proto_h, int proto_w,
    int orig_h, int orig_w) const {
    std::vector<cv::Mat> masks;
    masks.reserve(dets.size());
    if (dets.empty()) return masks;

    int proto_hw = proto_h * proto_w;

    // Letterbox geometry: same ratio/padding the Transformer used to fit the image into img_size.
    // (KR: letterbox 기하: Transformer 가 이미지를 img_size 에 맞출 때 쓴 ratio/padding 과 동일.)
    float r = std::min(static_cast<float>(img_size_) / orig_h,
                       static_cast<float>(img_size_) / orig_w);
    int new_h = static_cast<int>(std::round(orig_h * r));
    int new_w = static_cast<int>(std::round(orig_w * r));
    float dh = (img_size_ - new_h) / 2.0f;
    float dw = (img_size_ - new_w) / 2.0f;
    int pad_top = static_cast<int>(dh);
    int pad_left = static_cast<int>(dw);

    // Wrap the prototype as a (proto_c, proto_h*proto_w) matrix so a coefficient row-vector
    // times the matrix yields one mask map per detection.
    // (KR: prototype 을 (proto_c, proto_h*proto_w) 행렬로 보고, 계수 행벡터 x 행렬로 탐지당 마스크 맵 하나 생성.)
    cv::Mat proto_mat(proto_c, proto_hw, CV_32F,
                      const_cast<float*>(proto.data()));

    for (const auto& d : dets) {
        // mask_lin = coeff (1 x proto_c) * proto (proto_c x proto_hw) -> (1 x proto_hw).
        // (KR: mask_lin = coeff(1 x proto_c) * proto(proto_c x proto_hw) -> (1 x proto_hw).)
        cv::Mat coeff(1, proto_c, CV_32F,
                      const_cast<float*>(d.mask_coeffs.data()));
        cv::Mat mask_lin = coeff * proto_mat;          // (1, proto_hw)
        cv::Mat mask = mask_lin.reshape(1, proto_h);   // (proto_h, proto_w)

        // sigmoid(mask) > 0.5 is equivalent to raw mask logit > 0.0 (matches process_mask_upsample.gt_(0.0)).
        // (KR: sigmoid(mask) > 0.5 는 raw mask logit > 0.0 과 동치(process_mask_upsample 의 gt_(0.0) 과 일치).)
        cv::Mat upsized;
        cv::resize(mask, upsized, cv::Size(img_size_, img_size_), 0, 0, cv::INTER_LINEAR);

        // Crop to the box (letterbox coords): zero out everything outside the predicted bbox.
        // (KR: box(letterbox 좌표)로 crop: 예측 bbox 바깥을 0 으로 만든다.)
        cv::Mat binm = cv::Mat::zeros(img_size_, img_size_, CV_8U);
        int bx1 = std::clamp(static_cast<int>(std::floor(d.x1)), 0, img_size_);
        int by1 = std::clamp(static_cast<int>(std::floor(d.y1)), 0, img_size_);
        int bx2 = std::clamp(static_cast<int>(std::ceil(d.x2)), 0, img_size_);
        int by2 = std::clamp(static_cast<int>(std::ceil(d.y2)), 0, img_size_);
        for (int y = by1; y < by2; ++y) {
            const float* mrow = upsized.ptr<float>(y);
            uint8_t* brow = binm.ptr<uint8_t>(y);
            for (int x = bx1; x < bx2; ++x) {
                if (mrow[x] > 0.0f) brow[x] = 255;
            }
        }

        // Strip the letterbox padding, then resize to the original image size and re-threshold > 0.5.
        // (KR: letterbox padding 제거 후 원본 이미지 크기로 resize, 다시 threshold > 0.5.)
        int crop_w = std::max(1, img_size_ - 2 * pad_left);
        int crop_h = std::max(1, img_size_ - 2 * pad_top);
        cv::Rect roi(pad_left, pad_top, std::min(crop_w, img_size_ - pad_left),
                     std::min(crop_h, img_size_ - pad_top));
        cv::Mat cropped = binm(roi);

        cv::Mat resized;
        cv::resize(cropped, resized, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);
        cv::Mat final_mask;
        cv::threshold(resized, final_mask, 127, 255, cv::THRESH_BINARY);
        masks.push_back(final_mask);
    }
    return masks;
}

void YoloSegDecoder::scale_to_original(std::vector<Detection>& dets,
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
