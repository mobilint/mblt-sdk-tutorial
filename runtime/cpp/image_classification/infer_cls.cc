// Image classification inference on Mobilint NPU.
//
// Preprocessing lives in preprocess() / preprocess_float(): resize 256 + centerCrop 224 + BGR2RGB,
// and for float also /255 + torch mean/std.
// Input mode via --input: uint8 feeds the fused-normalization MXQ; float feeds a normalized float tensor to the !uint8 MXQ.
//
// Usage:
//   ./infer-cls <model.mxq> <image_path> <labels_file> [--input uint8|float]
//
// Example:
//   ./infer-cls resnet50.mxq example.jpg imagenet_labels.txt   # ARIES / REGULUS regulus-rb

#include <qbruntime/qbruntime.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

std::vector<std::string> load_labels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream in(path);
    if (!in.is_open()) return labels;
    std::string line;
    while (std::getline(in, line)) {
        labels.push_back(line);
    }
    return labels;
}

// ResNet-50 spatial preprocessing: resize short edge to 256 + center crop 224x224 + BGR2RGB.
// Returns a uint8 HWC image; the uint8-input MXQ has normalization fused in.
cv::Mat preprocess(const cv::Mat& input) {
    cv::Mat img = input.clone();

    // Resize: short edge to 256, keep aspect ratio
    int short_edge = std::min(img.rows, img.cols);
    float scale = 256.0f / static_cast<float>(short_edge);
    int new_h = static_cast<int>(std::round(img.rows * scale));
    int new_w = static_cast<int>(std::round(img.cols * scale));
    cv::resize(img, img, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // CenterCrop: 224x224
    int x = (img.cols - 224) / 2;
    int y = (img.rows - 224) / 2;
    img = img(cv::Rect(x, y, 224, 224)).clone();

    // BGR -> RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    return img;
}

// Float preprocessing for the !uint8 MXQ: spatial preprocess + /255 + torch mean/std normalization.
// mean/std must match compilation/image_classification/convert_img_to_tensor.py. Returns HWC float.
std::vector<float> preprocess_float(const cv::Mat& input) {
    cv::Mat img = preprocess(input);  // resize + crop + BGR2RGB -> HWC uint8 RGB
    const int hw = img.rows * img.cols, c = img.channels();
    const float mean[3] = {0.485f, 0.456f, 0.406f};  // RGB
    const float stdv[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> out(static_cast<size_t>(hw) * c);
    const uint8_t* src = img.data;
    for (int i = 0; i < hw; ++i) {
        for (int ch = 0; ch < c; ++ch) {
            float v = static_cast<float>(src[i * c + ch]) / 255.0f;
            out[i * c + ch] = (v - mean[ch]) / stdv[ch];
        }
    }
    return out;
}

int main(int argc, char** argv) {
    // Positional: <model.mxq> <image_path> <labels_file>. Optional: --input uint8|float
    // uint8 : normalization fused into the MXQ.  float : preprocess_float normalizes for the !uint8 MXQ.
    std::vector<std::string> pos;
    std::string input_type = "uint8";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--input" && i + 1 < argc) {
            input_type = argv[++i];
        } else {
            pos.push_back(a);
        }
    }
    if (pos.size() != 3 || (input_type != "uint8" && input_type != "float")) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.mxq> <image_path> <labels_file> [--input uint8|float]\n";
        return 1;
    }

    const std::string mxq_path = pos[0];
    const std::string image_path = pos[1];
    const std::string labels_path = pos[2];
    std::cout << "Input mode: " << input_type << "\n";

    // 1) Load labels
    auto labels = load_labels(labels_path);
    if (labels.empty()) {
        std::cerr << "Failed to load labels: " << labels_path << "\n";
        return 1;
    }

    // 2) Load MXQ model onto NPU. Use single-core mode (Cluster0/Core0) so the
    // same binary handles both REGULUS single-mode mxq and ARIES multi-mode
    // mxq (inference_scheme="all" produced by model_compile.py).
    mobilint::StatusCode sc;
    auto acc = mobilint::Accelerator::create(sc);
    mobilint::ModelConfig mc;
    mc.setSingleCoreMode({mobilint::CoreId{mobilint::Cluster::Cluster0,
                                           mobilint::Core::Core0}});
    auto model = mobilint::Model::create(mxq_path, mc, sc);
    sc = model->launch(*acc);

    // Model input shape as declared in the MXQ (HWC order, matches mxqtool "Shape").
    const auto& in_shapes = model->getModelInputShape();
    for (size_t i = 0; i < in_shapes.size(); ++i) {
        std::cout << "Model input shape[" << i << "]: [";
        for (size_t j = 0; j < in_shapes[i].size(); ++j)
            std::cout << in_shapes[i][j] << (j + 1 < in_shapes[i].size() ? ", " : "");
        std::cout << "]\n";
    }

    // 3) Load image, preprocess (HWC, NPU-native layout), and run inference.
    //    uint8 : feed the cropped uint8 image directly (normalization fused in the MXQ).
    //    float : preprocess_float applies /255 + torch mean/std for the !uint8 MXQ.
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << "\n";
        return 1;
    }

    std::vector<std::vector<float>> output;
    auto t0 = std::chrono::high_resolution_clock::now();
    if (input_type == "float") {
        std::vector<float> finput = preprocess_float(img);
        output = model->infer({finput.data()}, sc);
    } else {
        cv::Mat input = preprocess(img);
        output = model->infer({input.data}, sc);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Inference time: " << infer_ms << " ms\n";

    // 5) Top-5 predictions
    auto& logits = output[0];
    std::vector<int> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + 5, indices.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });

    std::cout << "\nTop-5 predictions:\n";
    for (int i = 0; i < 5; ++i) {
        int idx = indices[i];
        std::string name = (idx < static_cast<int>(labels.size())) ? labels[idx] : "unknown";
        std::cout << "  " << idx << " " << name << " (" << logits[idx] << ")\n";
    }

    // 6) Cleanup
    model->dispose();

    return 0;
}
