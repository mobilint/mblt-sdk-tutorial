// Image classification inference on Mobilint NPU.
//
// Preprocessing: resize(256) + centerCrop(224) + BGR2RGB (done in code)
// Normalization: fused into MXQ model (fuseIntoFirstLayer)
// Input: uint8 cropped image
//
// Usage:
//   ./infer-cls <model.mxq> <image_path> <labels_file>
//
// Example:
//   ./infer-cls resnet50.mxq example.jpg imagenet_labels.txt

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

// ResNet-50 preprocessing: resize short edge to 256 + center crop 224x224 + BGR2RGB
// Normalization (mean/std) is fused into the MXQ model.
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

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model.mxq> <image_path> <labels_file>\n";
        return 1;
    }

    const std::string mxq_path = argv[1];
    const std::string image_path = argv[2];
    const std::string labels_path = argv[3];

    // 1) Load labels
    auto labels = load_labels(labels_path);
    if (labels.empty()) {
        std::cerr << "Failed to load labels: " << labels_path << "\n";
        return 1;
    }

    // 2) Load MXQ model onto NPU
    mobilint::StatusCode sc;
    auto acc = mobilint::Accelerator::create(sc);
    auto model = mobilint::Model::create(mxq_path, sc);
    sc = model->launch(*acc);

    auto info = model->getInputBufferInfo()[0];
    std::cout << "Model input: " << info.original_height << "x"
              << info.original_width << "x" << info.original_channel << "\n";

    // 3) Load and preprocess image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << "\n";
        return 1;
    }
    cv::Mat input = preprocess(img);

    // 4) Run NPU inference (uint8 input, normalization fused in MXQ)
    auto t0 = std::chrono::high_resolution_clock::now();
    auto output = model->infer({input.data}, sc);
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
