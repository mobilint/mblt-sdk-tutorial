#include "preprocessor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

PreprocessedImage Preprocessor::transform_uint8(const cv::Mat& input, int input_height, int input_width,
                                                bool channel_last) {
  if (input.empty()) {
    throw std::invalid_argument("Input image is empty.");
  }
  if (input.channels() != 3) {
    throw std::invalid_argument("Input image must have three channels.");
  }

  const float scale =
      std::min(static_cast<float>(input_height) / input.rows, static_cast<float>(input_width) / input.cols);
  const int resized_width = static_cast<int>(std::round(input.cols * scale));
  const int resized_height = static_cast<int>(std::round(input.rows * scale));

  cv::Mat image;
  if (input.cols != resized_width || input.rows != resized_height) {
    cv::resize(input, image, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_LINEAR);
  } else {
    image = input.clone();
  }

  const int pad_height = input_height - resized_height;
  const int pad_width = input_width - resized_width;
  const int top = pad_height / 2;
  const int bottom = pad_height - top;
  const int left = pad_width / 2;
  const int right = pad_width - left;
  cv::copyMakeBorder(image, image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  if (!image.isContinuous()) {
    image = image.clone();
  }

  PreprocessedImage result;
  result.letterbox = {input_height, input_width, top, bottom, left, right};
  result.data.resize(static_cast<size_t>(input_height) * input_width * 3);
  if (channel_last) {
    std::memcpy(result.data.data(), image.data, result.data.size());
  } else {
    const uint8_t* source = image.ptr<uint8_t>(0);
    const int pixels = input_height * input_width;
    for (int channel = 0; channel < 3; ++channel) {
      for (int index = 0; index < pixels; ++index) {
        result.data[channel * pixels + index] = source[index * 3 + channel];
      }
    }
  }
  return result;
}
