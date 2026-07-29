// End-to-end YOLO26 semantic-segmentation inference on a Mobilint NPU.
// Pipeline: letterbox -> uint8 MXQ inference -> 19-class argmax
// -> letterbox removal -> source-size restoration -> Cityscapes overlay.

#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <qbruntime/qbruntime.h>

#include "preprocessor.h"
#include "semantic_postprocess.h"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.mxq> <image_path> <output_path>\n";
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    const std::string output_path = argv[3];

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << "\n";
        return 1;
    }

    mobilint::StatusCode status;
    auto accelerator = mobilint::Accelerator::create(status);
    if (!status) {
        std::cerr << "Failed to create accelerator.\n";
        return 1;
    }

    mobilint::ModelConfig model_config;
    model_config.setSingleCoreMode(
        {mobilint::CoreId{mobilint::Cluster::Cluster0,
                          mobilint::Core::Core0}});
    auto model = mobilint::Model::create(model_path, model_config, status);
    if (!status) {
        std::cerr << "Failed to load model: " << model_path << "\n";
        return 1;
    }
    status = model->launch(*accelerator);
    if (!status) {
        std::cerr << "Failed to launch model.\n";
        return 1;
    }

    try {
        const auto& input_shapes = model->getModelInputShape();
        if (input_shapes.size() != 1 || input_shapes[0].size() != 3) {
            throw std::runtime_error(
                "Expected one three-dimensional model input.");
        }
        const std::vector<int64_t> input_shape(
            input_shapes[0].begin(), input_shapes[0].end());
        const bool channel_last = input_shape.back() == 3;
        const bool channel_first = input_shape.front() == 3;
        if (!channel_last && !channel_first) {
            throw std::runtime_error(
                "Could not determine the model input layout.");
        }
        const int input_height = static_cast<int>(
            channel_last ? input_shape[0] : input_shape[1]);
        const int input_width = static_cast<int>(
            channel_last ? input_shape[1] : input_shape[2]);

        std::cout << "Model input shape: ["
                  << input_shape[0] << ", " << input_shape[1] << ", "
                  << input_shape[2] << "]\n";
        std::cout << "Image size: " << image.cols << "x" << image.rows << "\n";

        PreprocessedImage input = Preprocessor::transform_uint8(
            image, input_height, input_width, channel_last);
        std::vector<mobilint::NDArray<uint8_t>> inputs{
            mobilint::NDArray<uint8_t>(input.data.data(), input_shape)};

        const auto start = std::chrono::high_resolution_clock::now();
        std::vector<mobilint::NDArray<float>> outputs =
            channel_last ? model->infer(inputs, status)
                         : model->inferCHW(inputs, status);
        const auto end = std::chrono::high_resolution_clock::now();
        if (!status) {
            throw std::runtime_error("NPU inference failed.");
        }
        if (outputs.size() != 1) {
            throw std::runtime_error(
                "Semantic segmentation expects one output tensor.");
        }
        std::cout << "Inference time: "
                  << std::chrono::duration<double, std::milli>(end - start)
                         .count()
                  << " ms\n";

        const auto& output_shape = outputs[0].shape();
        std::cout << "Raw MXQ output shape: [";
        for (size_t index = 0; index < output_shape.size(); ++index) {
            std::cout << output_shape[index]
                      << (index + 1 < output_shape.size() ? ", " : "");
        }
        std::cout << "]\n";

        cv::Mat class_map = postprocess_semantic(
            outputs[0], input.letterbox, image.size());
        cv::Mat result = visualize_semantic(image, class_map);
        const std::filesystem::path output_file(output_path);
        if (!output_file.parent_path().empty()) {
            std::filesystem::create_directories(output_file.parent_path());
        }
        if (!cv::imwrite(output_path, result)) {
            throw std::runtime_error(
                "Failed to save output image: " + output_path);
        }
        std::cout << "Result saved to: " << output_path << "\n";
    } catch (const std::exception& error) {
        model->dispose();
        std::cerr << error.what() << "\n";
        return 1;
    }

    model->dispose();
    return 0;
}
