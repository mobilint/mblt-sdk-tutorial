// =============================================================================
// runner.h - NPU 실행 래퍼 (NPU runner) + .npy 작성 유틸
// =============================================================================
// qbruntime Accelerator/Model 을 감싸는 클래스. 타겟 보드(ARM64)에서만 NPU 접근.
//
// Wraps qbruntime Accelerator/Model. NPU access only on the target board (ARM64).
// =============================================================================
#pragma once
#include <qbruntime/qbruntime.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class NPURunner {
public:
    explicit NPURunner(const std::string& model_path);
    ~NPURunner();

    // float 입력 추론. HWC float 배열을 받아 출력 텐서 벡터 반환.
    std::vector<std::vector<float>> infer(std::unique_ptr<float[]> input);

    // uint8 입력 추론 (uint8-input compiled MXQ).
    std::vector<std::vector<float>> infer_uint8(std::unique_ptr<uint8_t[]> input);

    // 모델 입력 shape: {height, width, channel}
    std::vector<int> get_input_shape() const;

private:
    mobilint::StatusCode sc_;
    std::unique_ptr<mobilint::Accelerator> acc_;
    std::unique_ptr<mobilint::Model> model_;
};

// .npy (numpy 표준) 단일 텐서 저장. zip / 외부 라이브러리 없음.
// 포맷: magic(\x93NUMPY) + ver(1.0) + JSON 헤더 + binary float32.
// path: 출력 파일 경로
// data: 평면 float32 배열 (row-major)
// shape: 차원 순서 그대로 (예: {1, 64, 80, 80})
void save_npy(const std::string& path, const float* data,
              const std::vector<size_t>& shape);
