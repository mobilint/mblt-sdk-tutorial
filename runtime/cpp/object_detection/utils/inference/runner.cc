// Implementation of NPURunner and save_npy() declared in runner.h.
// NPURunner wraps qbruntime Accelerator/Model lifecycle (create, launch, infer, dispose).
// save_npy() writes a single float32 tensor to disk in NumPy v1.0 format without external libraries.
//
// (KR) runner.h 에 선언된 NPURunner 와 save_npy() 구현.
// NPURunner 는 qbruntime Accelerator/Model 의 생명주기(생성, 실행, 추론, 해제)를 래핑한다.
// save_npy() 는 외부 라이브러리 없이 float32 텐서를 NumPy v1.0 포맷으로 저장한다.
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

    // Single-core mode (Cluster0/Core0) works for both ARIES multi-mode MXQ and REGULUS single-mode MXQ.
    // (KR: 단일 코어 모드는 ARIES 멀티모드 MXQ 와 REGULUS 단일모드 MXQ 양쪽에서 동작한다.)
    mobilint::ModelConfig mc;
    mc.setSingleCoreMode({mobilint::CoreId{mobilint::Cluster::Cluster0,
                                           mobilint::Core::Core0}});
    model_ = mobilint::Model::create(model_path, mc, sc_);
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
    // Must use inferCHW because transform_uint8 produces CHW layout.
    // The plain infer(uint8_t*) overload assumes NHWC, which causes the NPU to misinterpret pixels and produce garbage output.
    // (KR: transform_uint8 출력이 CHW 이므로 inferCHW 를 써야 한다. infer(uint8_t*) 는 NHWC 를 가정해 garbage 결과를 낸다.)
    return model_->inferCHW({input.get()}, sc_);
}

std::vector<int> NPURunner::get_input_shape() const {
    auto info = model_->getInputBufferInfo()[0];
    return {static_cast<int>(info.original_height),
            static_cast<int>(info.original_width),
            static_cast<int>(info.original_channel)};
}

// NumPy v1.0 .npy format: magic(6 B) + version(2 B) + header_len uint16-LE(2 B)
// + ASCII dict header padded to 16-byte alignment ending with '\n' + raw float32 data row-major.
// (KR: NumPy v1.0 .npy 포맷: magic(6 B) + version(2 B) + header_len uint16-LE(2 B)
// + 16바이트 정렬 ASCII dict 헤더('\n' 종료) + float32 raw 데이터 row-major.)
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

    // Prefix is 10 bytes (magic 6 + ver 2 + header_len 2); pad header to 16-byte alignment then append '\n'.
    // (KR: 접두사 10바이트(magic 6 + ver 2 + header_len 2); 헤더를 16바이트 정렬 후 '\n' 추가.)
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
