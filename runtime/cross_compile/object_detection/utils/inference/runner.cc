// =============================================================================
// runner.cc - NPU 실행 래퍼 + .npy 작성 유틸 구현
// =============================================================================
#include "runner.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

NPURunner::NPURunner(const std::string& model_path) {
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Model file not found: " + model_path);
    }
    acc_ = mobilint::Accelerator::create(sc_);
    if (!sc_) throw std::runtime_error("Failed to create accelerator");

    model_ = mobilint::Model::create(model_path, sc_);
    if (!sc_) throw std::runtime_error("Failed to load model: " + model_path);

    sc_ = model_->launch(*acc_);
    if (!sc_) throw std::runtime_error("Failed to launch model");
}

NPURunner::~NPURunner() {
    if (model_) model_->dispose();
}

std::vector<std::vector<float>> NPURunner::infer(std::unique_ptr<float[]> input) {
    return model_->infer({input.get()}, sc_);
}

std::vector<std::vector<float>> NPURunner::infer_uint8(std::unique_ptr<uint8_t[]> input) {
    // transform_uint8 의 출력이 CHW 라 inferCHW 호출이 필요하다.
    // 일반 infer(uint8_t*) 는 NHWC layout 을 가정하므로 (qbruntime model.h L598-615),
    // CHW 데이터를 그쪽으로 보내면 NPU 가 잘못된 픽셀로 해석해 garbage 결과를 낸다.
    return model_->inferCHW({input.get()}, sc_);
}

std::vector<int> NPURunner::get_input_shape() const {
    auto info = model_->getInputBufferInfo()[0];
    return {static_cast<int>(info.original_height),
            static_cast<int>(info.original_width),
            static_cast<int>(info.original_channel)};
}

// =============================================================================
// .npy writer (numpy v1.0 spec)
// =============================================================================
// magic: \x93NUMPY  (6 bytes)
// version: \x01\x00 (2 bytes)
// header_len: uint16 little-endian (2 bytes)
// header: ASCII dict literal, padded with spaces, ends with '\n'
// data: raw binary, row-major
void save_npy(const std::string& path, const float* data,
              const std::vector<size_t>& shape) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Failed to open .npy: " + path);

    static const char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    ofs.write(magic, 6);
    const char ver[2] = {'\x01', '\x00'};
    ofs.write(ver, 2);

    std::ostringstream oss;
    oss << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (shape.size() == 1 || i + 1 < shape.size()) oss << ", ";
    }
    oss << "), }";
    std::string header = oss.str();

    // total prefix = 10 bytes (magic 6 + ver 2 + header_len 2). header 가 16-byte
    // 정렬되도록 공백 패딩 + '\n'.
    size_t total = 10 + header.size() + 1;
    size_t pad = (16 - (total % 16)) % 16;
    header.append(pad, ' ');
    header.push_back('\n');

    uint16_t header_len = static_cast<uint16_t>(header.size());
    ofs.write(reinterpret_cast<const char*>(&header_len), 2);
    ofs.write(header.data(), header.size());

    size_t numel = 1;
    for (size_t d : shape) numel *= d;
    ofs.write(reinterpret_cast<const char*>(data), numel * sizeof(float));
}
