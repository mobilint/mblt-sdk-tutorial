// Implementation of YoloDecoder declared in decode.h.
// Performs DFL box decoding, sigmoid class scoring, and class-offset greedy NMS on NPU outputs.
#include "decode.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static inline float invsigmoid(float x) { return -std::log(1.0f / x - 1.0f); }

YoloDecoder::YoloDecoder(int nc, int nl, int img_size, int reg_max, float conf_thres, float iou_thres, int max_det)
    : nc_(nc),
      nl_(nl),
      img_size_(img_size),
      reg_max_(reg_max),
      conf_thres_(conf_thres),
      iou_thres_(iou_thres),
      max_det_(max_det) {
  invconf_ = invsigmoid(conf_thres_);

  // Strides follow 2^(3+i): [8, 16, 32, ...] for P3/P4/P5 feature maps.
  strides_.resize(nl_);
  for (int i = 0; i < nl_; ++i) {
    strides_[i] = 1 << (3 + i);
  }
  // Anchor centers are offset by 0.5 to place them at grid cell centers.
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

// Classifies flat NPU output tensors by stride into box (reg_max*4 * H*W) and cls (nc * H*W) groups.
struct StagedTensor {
  const float* data;
  int channels;
  int hw;
  int stride;
  bool is_cls;
};

static std::vector<StagedTensor> stage_outputs(const std::vector<mobilint::NDArray<float>>& raw, int nc, int reg_max,
                                               int img_size, const std::vector<int>& strides) {
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
  // Sort ascending by stride (8, 16, 32) to match the anchor flattening order in the constructor.
  auto by_stride = [](const StagedTensor& a, const StagedTensor& b) { return a.stride < b.stride; };
  std::sort(det_tensors.begin(), det_tensors.end(), by_stride);
  std::sort(cls_tensors.begin(), cls_tensors.end(), by_stride);

  if (det_tensors.size() != cls_tensors.size()) {
    throw std::runtime_error("decode: det/cls tensor count mismatch (det=" + std::to_string(det_tensors.size()) +
                             " cls=" + std::to_string(cls_tensors.size()) + ")");
  }
  std::vector<StagedTensor> ordered;
  ordered.reserve(det_tensors.size() * 2);
  for (size_t i = 0; i < det_tensors.size(); ++i) {
    ordered.push_back(det_tensors[i]);
    ordered.push_back(cls_tensors[i]);
  }
  return ordered;
}

// Computes IoU between two boxes given in xyxy format.
static inline float iou_xyxy(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2) {
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
    const std::vector<mobilint::NDArray<float>>& raw_outputs) const {
  const int total_anchors = static_cast<int>(anchors_.size());
  if (total_anchors == 0) return {};

  auto staged = stage_outputs(raw_outputs, nc_, reg_max_, img_size_, strides_);
  if (staged.empty()) return {};

  // Build per-anchor access structs; staged is ordered [det0, cls0, det1, cls1, ...].
  // Outputs are HWC (channel-last, as the MXQ declares): value(spatial, channel) lives at
  // base[spatial * num_channels + channel]. So each anchor only needs its base pointers and
  // its spatial index; the channel stride is the tensor's channel count (reg_max*4 or nc).
  struct AnchorAccess {
    const float* box_base;  // start of the box tensor, HWC layout (hw x reg_max*4)
    const float* cls_base;  // start of the cls tensor, HWC layout (hw x nc)
    int local;              // spatial index within this stride's grid (0..hw-1)
  };
  std::vector<AnchorAccess> access(total_anchors);

  int anchor_idx = 0;
  for (size_t st = 0; st < staged.size(); st += 2) {
    const auto& det = staged[st];
    const auto& cls = staged[st + 1];
    for (int i = 0; i < det.hw; ++i) {
      access[anchor_idx] = {det.data, cls.data, i};
      ++anchor_idx;
    }
  }

  // Invariant: every anchor slot in `access` must be populated before the pre-filter loop
  // dereferences it. stage_outputs() only checks that det/cls tensor counts match, so a
  // partial input (one stride pair missing) could leave trailing anchor slots uninitialized.
  if (anchor_idx != total_anchors) {
    throw std::runtime_error("YoloDecoder::decode: NPU outputs do not cover all anchors (" +
                             std::to_string(anchor_idx) + " of " + std::to_string(total_anchors) + " populated)");
  }

  // Pre-filter: keep only anchors whose max cls logit exceeds invconf_ (cheap logit-space threshold).
  std::vector<int> active;
  active.reserve(total_anchors);
  for (int a = 0; a < total_anchors; ++a) {
    // HWC: this anchor's nc class logits are contiguous at cls_base + local*nc.
    const float* cls = access[a].cls_base + static_cast<size_t>(access[a].local) * nc_;
    float max_logit = cls[0];
    for (int c = 1; c < nc_; ++c) {
      if (cls[c] > max_logit) max_logit = cls[c];
    }
    if (max_logit > invconf_) active.push_back(a);
  }
  if (active.empty()) return {};

  // DFL decode + sigmoid(cls): convert passing anchors to (x1,y1,x2,y2,conf,cls) detections.
  std::vector<Detection> dets;
  dets.reserve(active.size() * 2);

  const int box_ch = reg_max_ * 4;  // box tensor channel count (HWC last dim)

  std::vector<float> dfl_logits(reg_max_);
  std::vector<float> dfl_softmax(reg_max_);

  for (int a : active) {
    const auto& acc = access[a];
    // HWC: this anchor's box and cls channels are contiguous rows at base + local*channels.
    const float* box = acc.box_base + static_cast<size_t>(acc.local) * box_ch;
    const float* cls = acc.cls_base + static_cast<size_t>(acc.local) * nc_;

    // DFL softmax over reg_max bins for each of 4 sides (left, top, right, bottom).
    float dist[4];
    for (int side = 0; side < 4; ++side) {
      float maxv = -std::numeric_limits<float>::infinity();
      for (int r = 0; r < reg_max_; ++r) {
        float v = box[side * reg_max_ + r];
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
    float cx = anchors_[a].first;
    float cy = anchors_[a].second;
    float st = stride_per_anchor_[a];
    float x1 = (cx - dist[0]) * st;
    float y1 = (cy - dist[1]) * st;
    float x2 = (cx + dist[2]) * st;
    float y2 = (cy + dist[3]) * st;

    // Emit one Detection per class whose sigmoid score exceeds conf_thres.
    for (int c = 0; c < nc_; ++c) {
      float logit = cls[c];
      if (logit <= invconf_) continue;
      float conf = sigmoid(logit);
      if (conf <= conf_thres_) continue;
      dets.push_back({x1, y1, x2, y2, conf, c});
    }
  }
  if (dets.empty()) return {};

  // Cap candidates at 30000 before NMS to bound worst-case O(n^2) cost (matches ultralytics default).
  constexpr int max_pre = 30000;
  if (static_cast<int>(dets.size()) > max_pre) {
    std::partial_sort(dets.begin(), dets.begin() + max_pre, dets.end(),
                      [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
    dets.resize(max_pre);
  } else {
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
  }

  // Greedy NMS with per-class coordinate offset (max_wh=7680) so boxes of different classes never suppress each other.
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

void YoloDecoder::scale_to_original(std::vector<Detection>& dets, int img_size, int orig_h, int orig_w) {
  float r = std::min(static_cast<float>(img_size) / orig_h, static_cast<float>(img_size) / orig_w);
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
