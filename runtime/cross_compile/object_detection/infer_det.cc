// Object detection inference on Mobilint NPU with post-processing and visualization.
//
// Preprocessing: letterbox + BGR2RGB (done in code)
// Normalization: fused into MXQ model (Uint8InputConfig)
// Input: uint8 letterboxed image
//
// Usage:
//   ./infer-det <model.mxq> <image_path> <output_path>
//
// Example:
//   ./infer-det yolov9m.mxq example.jpg result.jpg

#include <qbruntime/qbruntime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

static const std::vector<std::string> COCO_LABELS = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};
static const int NUM_CLASSES = 80;
static const int REG_MAX = 16;

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

// Letterbox: aspect-ratio-preserving resize + pad with 114
// Returns letterboxed image and the ratio/padding for coordinate reverse mapping
cv::Mat letterbox(const cv::Mat& img, int target_size, float& ratio, float& pad_x, float& pad_y) {
    int h = img.rows, w = img.cols;
    ratio = std::min(static_cast<float>(target_size) / h,
                     static_cast<float>(target_size) / w);
    int new_w = static_cast<int>(std::round(w * ratio));
    int new_h = static_cast<int>(std::round(h * ratio));

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int dw = target_size - new_w;
    int dh = target_size - new_h;
    int top = dh / 2, bottom = dh - top;
    int left = dw / 2, right = dw - left;
    pad_x = static_cast<float>(left);
    pad_y = static_cast<float>(top);

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return padded;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float inverse_sigmoid(float x) {
    return -std::log(1.0f / x - 1.0f);
}

float dfl_decode(const float* data) {
    float max_val = *std::max_element(data, data + REG_MAX);
    float sum = 0.0f;
    float vals[REG_MAX];
    for (int i = 0; i < REG_MAX; ++i) {
        vals[i] = std::exp(data[i] - max_val);
        sum += vals[i];
    }
    float result = 0.0f;
    for (int i = 0; i < REG_MAX; ++i) {
        result += (vals[i] / sum) * i;
    }
    return result;
}

std::vector<int> generate_grid(int grid_h, int grid_w) {
    std::vector<int> grid;
    grid.reserve(grid_h * grid_w * 2);
    for (int y = 0; y < grid_h; ++y) {
        for (int x = 0; x < grid_w; ++x) {
            grid.push_back(x);
            grid.push_back(y);
        }
    }
    return grid;
}

void decode_scale(const std::vector<float>& box_tensor,
                  const std::vector<float>& cls_tensor,
                  const std::vector<int>& grid, int stride,
                  int grid_h, int grid_w, float inv_conf_thres,
                  std::vector<Detection>& dets) {
    int num_cells = grid_h * grid_w;
    for (int i = 0; i < num_cells; ++i) {
        int best_cls = -1;
        float best_logit = inv_conf_thres;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float logit = cls_tensor[i * NUM_CLASSES + c];
            if (logit > best_logit) {
                best_logit = logit;
                best_cls = c;
            }
        }
        if (best_cls < 0) continue;

        const float* box_base = &box_tensor[i * 4 * REG_MAX];
        float left   = dfl_decode(box_base + 0 * REG_MAX);
        float top    = dfl_decode(box_base + 1 * REG_MAX);
        float right  = dfl_decode(box_base + 2 * REG_MAX);
        float bottom = dfl_decode(box_base + 3 * REG_MAX);

        float gx = static_cast<float>(grid[i * 2 + 0]);
        float gy = static_cast<float>(grid[i * 2 + 1]);

        float x1 = (gx - left + 0.5f) * stride;
        float y1 = (gy - top + 0.5f) * stride;
        float x2 = (gx + right + 0.5f) * stride;
        float y2 = (gy + bottom + 0.5f) * stride;

        dets.push_back({x1, y1, x2, y2, sigmoid(best_logit), best_cls});
    }
}

float iou(const Detection& a, const Detection& b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

std::vector<Detection> nms(std::vector<Detection>& dets, float iou_thres) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> result;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!suppressed[j] && dets[i].class_id == dets[j].class_id &&
                iou(dets[i], dets[j]) > iou_thres) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

void rescale_boxes(std::vector<Detection>& dets, int img_w, int img_h,
                   float ratio, float pad_x, float pad_y) {
    for (auto& d : dets) {
        d.x1 = std::clamp((d.x1 - pad_x) / ratio, 0.0f, static_cast<float>(img_w));
        d.y1 = std::clamp((d.y1 - pad_y) / ratio, 0.0f, static_cast<float>(img_h));
        d.x2 = std::clamp((d.x2 - pad_x) / ratio, 0.0f, static_cast<float>(img_w));
        d.y2 = std::clamp((d.y2 - pad_y) / ratio, 0.0f, static_cast<float>(img_h));
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model.mxq> <image_path> <output_path>\n";
        return 1;
    }

    const std::string mxq_path = argv[1];
    const std::string image_path = argv[2];
    const std::string output_path = argv[3];
    const float conf_thres = 0.25f;
    const float iou_thres = 0.45f;
    const float inv_conf_thres = inverse_sigmoid(conf_thres);
    const int model_size = 640;
    const int nl = 3;
    const std::vector<int> strides = {8, 16, 32};

    // 1) Load MXQ model onto NPU
    mobilint::StatusCode sc;
    auto acc = mobilint::Accelerator::create(sc);
    auto model = mobilint::Model::create(mxq_path, sc);
    sc = model->launch(*acc);

    auto info = model->getInputBufferInfo()[0];
    std::cout << "Model input: " << info.original_height << "x"
              << info.original_width << "x" << info.original_channel << "\n";

    // 2) Load image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << "\n";
        return 1;
    }
    int img_h = img.rows, img_w = img.cols;
    std::cout << "Image size: " << img_w << "x" << img_h << "\n";

    // 3) Preprocess: BGR->RGB + letterbox
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    float ratio, pad_x, pad_y;
    cv::Mat input = letterbox(rgb, model_size, ratio, pad_x, pad_y);

    // 4) Run NPU inference (uint8 input, normalization fused in MXQ)
    auto t0 = std::chrono::high_resolution_clock::now();
    auto output = model->infer({input.data}, sc);
    auto t1 = std::chrono::high_resolution_clock::now();
    double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Inference time: " << infer_ms << " ms\n";

    // 5) Post-process: decode multi-scale DFL outputs
    std::vector<Detection> dets;
    for (int i = 0; i < nl; ++i) {
        int cls_idx = static_cast<int>(output.size()) - 1 - 2 * i;
        int box_idx = cls_idx - 1;
        int grid_h = model_size / strides[i];
        int grid_w = model_size / strides[i];
        auto grid = generate_grid(grid_h, grid_w);
        decode_scale(output[box_idx], output[cls_idx], grid, strides[i],
                     grid_h, grid_w, inv_conf_thres, dets);
    }

    // 6) NMS + rescale to original image coordinates
    dets = nms(dets, iou_thres);
    rescale_boxes(dets, img_w, img_h, ratio, pad_x, pad_y);
    std::cout << "Detections: " << dets.size() << "\n";

    // 7) Draw bounding boxes on original image (BGR)
    for (const auto& d : dets) {
        int x1 = static_cast<int>(d.x1), y1 = static_cast<int>(d.y1);
        int x2 = static_cast<int>(d.x2), y2 = static_cast<int>(d.y2);
        int pct = static_cast<int>(d.confidence * 100);
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2),
                      cv::Scalar(0, 255, 0), 2);
        std::string label = COCO_LABELS[d.class_id] + " " + std::to_string(pct) + "%";
        cv::putText(img, label, cv::Point(x1, y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        std::cout << "  " << COCO_LABELS[d.class_id] << " " << pct << "% "
                  << "[" << x1 << "," << y1 << "," << x2 << "," << y2 << "]\n";
    }

    // 8) Save result image
    cv::imwrite(output_path, img);
    std::cout << "Result saved to: " << output_path << "\n";

    // 9) Cleanup
    model->dispose();

    return 0;
}
