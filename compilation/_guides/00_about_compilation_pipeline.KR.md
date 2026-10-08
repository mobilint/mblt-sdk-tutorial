# 컴파일 파이프라인 개요

이 문서는 Mobilint qbcompiler를 사용한 모델 컴파일 파이프라인의 전체 흐름을 설명합니다.

## 개요

PyTorch, TensorFlow, ONNX 등의 모델은 GPU/CPU에서 추론하도록 설계되어 있습니다.
이러한 모델을 Mobilint NPU에서 실행하려면, NPU가 이해할 수 있는 형태로 변환(컴파일)해야 합니다.

qbcompiler는 다양한 프레임워크의 모델 변환을 지원합니다.

원본 모델의 형식은 `backend` 파라미터로 지정합니다(대소문자 구분 없음):

| backend | 입력 형식 | 이 저장소의 예시 |
| --------- | ---------- | ---------- |
| `"onnx"` (기본값) | ONNX 파일 경로 | image_classification (`resnet50.onnx`) |
| `"torch"` | PyTorch 모델 객체(`torch.nn.Module`), Hugging Face 모델 포함 | llm, bert, stt, vlm, mask_generation |
| `"tf"` (`"tensorflow"`, `"keras"`도 가능) | TensorFlow SavedModel 디렉터리, Keras `.keras` / `.h5` 파일, frozen GraphDef `.pb` | - |
| `"tflite"` | `.tflite` 파일 경로 | - |
| `"torchscript"` | `torch.jit.save`로 저장한 아카이브(먼저 ONNX로 내보내므로 `feed_dict` 필요) | - |

그 밖의 값은 `ValueError`를 발생시킵니다.
기존 `.mblt` 파일 경로를 넘기면 `backend`와 관계없이 파싱을 건너뜁니다(아래 참고).

```python
from qbcompiler import mblt_compile, mxq_compile

# ONNX 모델을 변환하는 경우 (image_classification)
mxq_compile(model="./resnet50.onnx", backend="onnx", target_device="aries-rb", ...)

# PyTorch 모델을 변환하는 경우 (llm): 불러온 모델 객체와 예시 입력을 넘김
mblt_compile(model=model, backend="torch", target_device="aries-rb", feed_dict=feed_dict, ...)
```

### Hugging Face 모델의 일부(part)만 파싱하기

여러 컴포넌트로 구성된 모델(STT encoder/decoder, VLM vision/language)은 `backend="torch"`로 파트 단위로 파싱합니다.
qbcompiler는 지원하는 아키텍처마다 파트를 선언해 두며, `qbcompiler.model_dict.parser.patcher.parts`와 `qbcompiler.model_dict.parser.backend.torch.input_capture`의 세 헬퍼로 준비합니다.

| API | 역할 |
| --- | --- |
| `load_for_part(model_id, part, *, dtype=None, device=None, revision=None, trust_remote_code=False)` | 해당 파트에 대해 파서가 기대하는 클래스로 체크포인트를 불러와 `eval()` 모드로 반환 |
| `prepare_part(model, part)` | 입력 캡처 전에 필요한 파트별 변경을 적용하고, 입력을 캡처할 모듈을 반환 |
| `capture_forward_inputs(module, *, to_cpu=True, ...)` | 실제 forward 또는 `generate()` 한 번 동안 `module.forward`의 인자를 기록하는 컨텍스트 매니저 |

캡처한 입력은 `feed_dict`로 넘기고, `mblt_compile()` / `mxq_compile()`은 `model_part`로 파트를 선택합니다.
`model_part_options`는 파트별 옵션을 전달합니다. 예를 들어 Whisper 디코더의 `{"last_token_only": True}`, Qwen3-VL 비전 인코더의 `{"side_inputs": True}`가 있습니다.

```python
import torch
from qbcompiler import mblt_compile
from qbcompiler.model_dict.parser.backend.torch.input_capture import capture_forward_inputs
from qbcompiler.model_dict.parser.patcher.parts import load_for_part, prepare_part

model = load_for_part("Qwen/Qwen3-VL-2B-Instruct", "vision", dtype=torch.float32, device="cuda")
with capture_forward_inputs(prepare_part(model, "vision"), to_cpu=False) as feed_dict:
    model.generate(**inputs, max_new_tokens=1)  # 이미지를 포함한 실제 forward 한 번

mblt_compile(
    model=model,
    model_part="vision",
    model_part_options={"side_inputs": True},
    backend="torch",
    target_device="aries-rb",
    mblt_save_path="./qwen3vl_encoder.mblt",
    feed_dict=dict(feed_dict),
    dynamic_axes={"images": [-1], "pos_embeds": [0], "cos": [-2], "sin": [-2]},
)
```

파트를 하나만 선언한 모델은 `model_part` 없이도 그 파트로 결정되고, 여러 개를 선언한 모델은 이름을 지정해야 합니다.
불러온 모델이 선언한 파트는 `qbcompiler.model_dict.parser.patcher.parts.available_parts(model)`로 확인할 수 있습니다.

### Dynamic axes와 multi-shape

- `dynamic_axes`는 런타임에 크기가 바뀌는 입력 축을 지정합니다. 예: LLM의 시퀀스 축, 비전 인코더의 patch 수 축.
- `multi_shape`는 대신 한 입력 축의 여러 크기를 한 번에 컴파일합니다. 예: `{"x": {"axis": 3, "values": [100, 200, 300]}}`.
  `mblt_compile()`은 크기마다 `.mblt`를 하나씩 만들고, `mxq_compile()`은 모든 크기를 하나의 `.mxq`로 묶습니다.
  `multi_shape`는 `feed_dict`가 필요하며 `dynamic_axes`와 함께 쓸 수 없습니다.

## 컴파일 파이프라인

컴파일 과정은 qbcompiler 내부적으로 **MBLT → MXQ** 두 단계를 거쳐 변환됩니다.

![Compilation Pipeline](../../assets/compilation_pipeline.png)

### MBLT (Mobilint Binary LayouT)

원본 모델의 연산 그래프와 가중치를 하드웨어 비의존적인 중간 형식으로 변환한 파일입니다.

### MXQ (Mobilint eXeQutable)

MBLT를 양자화하고 NPU 하드웨어에 최적화한 최종 배포 포맷입니다.
Mobilint NPU에서 직접 실행할 수 있는 `.mxq` 파일이 생성됩니다.

---

## 컴파일 방법

### `mxq_compile()`으로 한 번에 변환

대부분의 경우 `mxq_compile()`에 원본 모델을 넘기면
**MBLT → MXQ 변환이 내부적으로 자동 처리**됩니다.

사용자는 중간 MBLT 단계를 의식하지 않아도 됩니다.

```python
from qbcompiler import mxq_compile

mxq_compile(
    model="./resnet50.onnx",          # 원본 모델 경로
    target_device="aries-rb",         # 타깃 디바이스
    calib_data_path="./calib_data",   # calibration 데이터 경로
    save_path="./resnet50.mxq",       # MXQ 저장 경로
    backend="onnx",                   # 원본 모델 형식
    device="gpu",                     # 컴파일 장치 ("gpu" 또는 "cpu")
    inference_scheme="all",           # 추론 스킴. "all"은 single, multi, global4, global8 모두 지원
)
```

### 단계 분리: `mblt_compile()` → `mxq_compile()`

파싱과 양자화를 두 단계로 나눠 실행할 수도 있습니다.
`mblt_compile()`이 모델을 파싱해 `.mblt` 파일을 쓰고, 그 `.mblt` 경로를 `mxq_compile()`에 넘기면 파싱을 건너뛰고 양자화와 컴파일만 수행합니다.

멀티 컴포넌트 튜토리얼이 이 방식을 사용합니다.
VLM이나 STT처럼 하나의 모델이 여러 서브모델(encoder/decoder, vision/language)로 구성된 경우,
각 컴포넌트의 추론 호출 횟수나 양자화 설정이 다르므로 개별적으로 컴파일해야 합니다.
예를 들어 STT는 encoder가 1회 호출되는 동안 decoder는 토큰 수만큼 반복 호출됩니다.
VLM은 vision encoder가 이미지당 1회 호출되는 반면 language model은 토큰 수만큼 반복 호출됩니다.
단계를 나누면 하나의 `.mblt`를 여러 양자화 설정이나 추론 스킴에 재사용할 수도 있습니다.

```python
from qbcompiler import mblt_compile, mxq_compile

# 1. 파싱: 모델 -> MBLT
mblt_compile(
    model=model,                         # load_for_part("openai/whisper-small", "encoder", ...)로 불러온 모델
    model_part="encoder",
    mblt_save_path="./whisper_encoder.mblt",
    backend="torch",
    target_device="aries-rb",
    feed_dict={"input_features": input_features},
)

# 2. 양자화와 컴파일: MBLT -> MXQ
mxq_compile(
    model="./whisper_encoder.mblt",      # MBLT 경로를 넘김
    target_device="aries-rb",
    calib_data_path="./calib_data",
    save_path="./whisper_encoder.mxq",
    ...
)
```

`mxq_compile()`은 입력에 따라 호출을 나눕니다. 프레임워크 모델은 `mxq_compile_from_source()`가, 기존 `.mblt`는 `mxq_compile_from_mblt()`가 처리합니다.
두 함수를 직접 호출할 수도 있으며 양자화 인자는 `mxq_compile()`과 같습니다.
`.mblt` 파일 목록은 같은 그래프의 서로 다른 크기(예: `multi_shape` 파싱 결과)일 때만 받으며, 하나의 `.mxq`로 묶습니다.

> `mxq_compile()`의 `save_subgraph_type`과 `output_subgraph_path`는 시각화용 미리보기 `.mblt`만 내보냅니다.
> 이 미리보기는 `mxq_compile()`에 다시 넘길 수 없으므로, 실행 가능한 `.mblt`는 `mblt_compile()`로 만드세요.
>
> 멀티 컴포넌트 모델의 분리 컴파일에 대해서는
> [멀티 컴포넌트 모델 가이드](./03_about_multi_component.KR.md)를 참조하세요.

---

## 모델별 컴파일 경로 요약

| 모델 유형 | backend | MBLT 생성 | 비고 |
| ----------- | --------- | ----------- | ------ |
| Vision (classification, detection 등) | `onnx` | 확인용 `mblt_compile()`, `mxq_compile()`이 다시 파싱 | |
| LLM | `torch` | 명시적 (`mblt_compile`) | 4bit 시 SpinQuant 추가 |
| BERT | `torch` | 확인용 `mblt_compile()`, `mxq_compile()`이 다시 파싱 | |
| STT (Whisper) | `torch` | 명시적 (`mblt_compile(model_part=...)`) | `encoder` / `decoder` 파트 |
| VLM (Qwen3-VL) | `torch` | 명시적 (`mblt_compile(model_part=...)`) | `vision` / `language` 파트 |
| Mask generation (SAM2) | `torch` | 명시적 (`mblt_compile(model_part=...)`) | encoder/decoder 파트 |

## 다음 문서

- [양자화 설정 가이드](./01_about_quantization_config.KR.md) - `mxq_compile()`에 전달하는 config 옵션 상세 설명
- [Calibration 데이터 가이드](./02_about_calibration_data.KR.md) - calibration 데이터의 준비 및 포맷
- [멀티 컴포넌트 모델 가이드](./03_about_multi_component.KR.md) - VLM/STT 등 분리 컴파일이 필요한 모델
