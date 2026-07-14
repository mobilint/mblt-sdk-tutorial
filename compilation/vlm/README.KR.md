# 비전 언어 모델 (VLM) 컴파일

이 튜토리얼은 Mobilint `qbcompiler`로 비전 언어 모델(VLM)을 컴파일하는 방법을 설명합니다.

이 튜토리얼에서는 Qwen에서 개발한 최첨단 비전-언어 모델인 [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) 모델을 사용합니다. Qwen3-VL은 deepstack 비전 경로를 도입합니다. 비전 인코더는 image embeds와 함께 3개의 deepstack 특징 맵을 생성하며, 이는 디코더 초기 레이어에 주입됩니다. 따라서 캘리브레이션 데이터와 디코더 컴파일 모두 deepstack 텐서를 포함합니다.

이 튜토리얼의 양자화 설정은 벤치마크 최적 4B 구성을 사용합니다. 디코더는 4비트 가중치(value projection은 8비트로 승격), 임베딩 및 deepstack 입력에 대한 16비트 활성화, SpinR1/SpinR2 회전, OPTQ, weight-scale 탐색으로 컴파일됩니다. 인코더는 merger 및 deepstack merger `fc2` 레이어에 대한 16비트 활성화와 디코더의 SpinR1 행렬을 재사용하는 `head_out_ch_rotation`을 사용합니다.

## 개요

VLM 컴파일 과정은 세 가지 주요 단계로 구성됩니다:

1. **캘리브레이션 데이터 생성**: 양자화를 위한 캘리브레이션 데이터셋을 생성합니다.
2. **MBLT 컴파일**: 모델을 MBLT(Mobilint Binary LayouT) 형식으로 컴파일합니다.
3. **MXQ 컴파일**: 고급 양자화를 적용해 배포용 `.mxq` 형식으로 컴파일합니다.

이 워크플로에서는 **언어 모델**(디코더)과 **비전 인코더**를 각각 별도로 컴파일합니다.

컴파일이 끝나면 `mxq/` 디렉토리에는 런타임 단계(`prepare_model.py`)가 배포 가능한 자체 완결형 모델 폴더로 만들어 주는 컴파일 산출물이 정리됩니다.

## 사전 요구사항

시작하기 전에 다음 항목이 준비되어 있는지 확인하세요.

- Python 3.8 이상
- `qbcompiler` SDK 설치 (버전 `>= 1.0.1`)
- 선택 사항: 캘리브레이션 및 컴파일에 사용할 CUDA 지원 GPU
- 충분한 디스크 공간 (모델과 캘리브레이션 데이터를 포함해 약 20 GB)

### 필수 의존성 패키지 설치

컴파일에 필요한 Python 패키지를 설치해야 합니다:

```bash
pip install transformers==4.57.1 qwen-vl-utils==0.0.14 accelerate==1.13.0
```

### 캘리브레이션 이미지 다운로드

캘리브레이션 과정에서는 COCO 데이터셋 이미지를 사용합니다. 100장의 이미지를 자동으로 내려받는 스크립트가 제공됩니다.

```bash
python download_images.py
```

**이 작업의 내용:**

- Hugging Face Datasets를 사용해 COCO 2017 validation에서 이미지 100장 다운로드
- 이미지를 `224x224` 해상도로 자동 리사이즈
- 이미지를 JPEG 파일로 `images/` 디렉토리에 저장
- COCO 다운로드가 실패하면 합성 샘플 이미지로 대체

**출력:**

- `images/image_0000.jpg`부터 `images/image_0099.jpg`까지

캘리브레이션 스크립트는 `images/` 디렉토리의 모든 이미지를 자동으로 사용하고 다양한 프롬프트(상세 설명, 시각적 추론, 카운팅, 공간 이해 등)를 순환하여 캘리브레이션 다양성을 보장합니다.

## Stage 1: 캘리브레이션 데이터 생성

캘리브레이션 데이터는 양자화에 필수적이며, 컴파일러가 모델의 일반적인 활성화 범위를 이해하는 데 도움이 됩니다.

### Step 1.1: 캘리브레이션 데이터 생성

하나의 스크립트가 모델을 한 번만 로드해 언어 모델(디코더)과 비전 인코더 캘리브레이션 데이터를 모두 생성합니다:

```bash
python generate_calibration_data.py \
    --model-name Qwen/Qwen3-VL-4B-Instruct \
    --output-dir ./calibration_data \
    --num-samples 100 \
    --max-new-tokens 512
```

**매개변수:**

- `--model-name`: HuggingFace 모델 식별자
- `--output-dir`: 기준 디렉토리. 하위에 `language/`, `vision/`, `prefill/`, `decode/` 가 생성됩니다
- `--num-samples`: 캘리브레이션 샘플 수 (기본값: 사용 가능한 모든 이미지)
- `--max-new-tokens`: decode 생성 단계의 최대 생성 토큰 수
- `--intermediate-ratios`: 저장할 decode 프리픽스 비율 (1.0은 항상 추가됨)

**이 작업의 내용:**

- `images/` 폴더에서 모든 이미지 로드 후 20가지 다양한 프롬프트 유형 순환
- 언어: 이미지당 두 번의 패스를 실행합니다. Pass 1은 비전 특징이 병합된 후 디코더 prefill 입력(`inputs_embeds`와 3개 레이어의 `deepstack_visual_embeds`)을 캡처하고, Pass 2는 full generate로 decode 토큰 시퀀스를 수집합니다. 이후 prefill과 decode 샘플을 하나의 `language/` 디렉토리로 병합합니다
- 비전: 비전 인코더 픽셀 값을 NPU 레이아웃 `[H, W, 6]` 으로 리셰이프 (이미지 크기 224x224 고정)
- 캘리브레이션 데이터를 `.npy` 파일로 저장. `language/` 는 단일 `npy_files.json` 으로 인덱싱됩니다

**출력 구조:**

```text
calibration_data/
 language/
    prefill_000/{inputs_embeds.npy, deepstack_visual_embeds.npy}  # [1, seq_len, 2048], [3, seq_len, 2048]
    decode_000/{inputs_embeds.npy, deepstack_visual_embeds.npy}
    ...
    npy_files.json
 vision/
    sample_000/images.npy           # [H, W, 6]
    ...
    npy_files.txt
```

## Stage 2: MBLT 컴파일

MBLT(Mobilint Binary LayouT)는 모델 그래프와 가중치를 하드웨어 독립적인 방식으로 나타내는 중간 형식입니다.

### Step 2.1: 언어 모델을 MBLT로 컴파일

언어 모델(디코더)을 MBLT 형식으로 컴파일합니다. `--target-device` 는 필수입니다 (`aries-rb` 또는 `regulus-rb`):

```bash
# ARIES
python mblt_compile_language.py --target-device aries-rb

# REGULUS (2026-06 이후 고객)
python mblt_compile_language.py --target-device regulus-rb
```

**이 작업의 내용:**

- 샘플 생성 중 언어 모델 입력 캡처
- 시퀀스 길이 차원을 동적으로 표시 (가변 길이 입력용)
- NPU 호환 아키텍처 패치 적용:
  - **사전 캐시된 RoPE 임베딩**: 런타임 삼각 함수 연산 제거
  - **마지막 쿼리 슬라이싱**: 디코드 단계를 위해 최종 디코더 레이어 최적화
  - **상태 저장 KV 캐시 래퍼**: 효율적인 자기 회귀 생성 활성화
  - **동적 형태 처리**: 가변 시퀀스 길이 지원
- 3개 레이어의 `deepstack_visual_embeds` 를 `build_full_visual_embeds` 와 `visual_pos_masks` 로 전체 시퀀스 길이에 맞춰 패딩
- 캐시된 rotary embedding 을 위해 `position_ids` 를 처음 3개 mrope 축(t/h/w)으로 슬라이스
- 어텐션 연산자에 대한 동적 형태 구성
- `mblt_compile()` 로 MBLT 형식 export

**주요 변환:**

- 입력 임베딩 차원을 동적으로 표시: `[batch, seq_len, hidden_size]`
- `deepstack_visual_embeds` 의 시퀀스 길이 축을 동적으로 표시
- 가변 시퀀스를 위해 어텐션 마스크 및 위치 ID를 동적으로 표시
- 자기 회귀 생성을 위해 캐시 위치를 동적으로 표시
- 캡처된 position ID 로부터 RoPE 임베딩 사전 계산

**출력 파일:**

- `./mblt/Qwen3-VL-4B-Instruct_text_model.mblt`: MBLT 형식의 컴파일된 모델

### Step 2.2: 비전 인코더를 MBLT로 컴파일

비전 인코더를 MBLT 형식으로 컴파일합니다. `--target-device` 는 필수입니다 (`aries-rb` 또는 `regulus-rb`):

```bash
# ARIES
python mblt_compile_vision.py --target-device aries-rb

# REGULUS (2026-06 이후 고객)
python mblt_compile_vision.py --target-device regulus-rb
```

**이 작업의 내용:**

- 샘플 추론 중 비전 인코더 입력 캡처
- 픽셀 값을 NPU 호환 형식으로 재처리
- NPU 호환 아키텍처 패치 적용:
  - **3D2D 컨볼루션**: NPU 최적화를 위해 3D 컨볼루션을 2D로 변환
  - **분할 QKV 프로젝션**: 더 나은 병렬화를 위해 Query, Key, Value 프로젝션 분리
  - **사전 계산된 RoPE 임베딩**: 런타임 삼각 함수 연산 제거
  - **병합된 패치화 연산**: 메모리 전송 감소
- `mblt_compile()` 로 MBLT 형식 export

**주요 변환:**

- 픽셀 값을 HuggingFace 형식 `[num_patches, channels*patch_size^2]`에서 ARIES 형식 `[batch, channels*temporal, height, width]`로 재처리
- 3D 시간 컨볼루션을 2D 공간 컨볼루션으로 변환
- 병렬 실행을 위해 QKV 어텐션 프로젝션 분리
- 이미지 그리드 차원을 기반으로 RoPE 임베딩 사전 계산

**출력 파일:**

- `./mblt/Qwen3-VL-4B-Instruct_vision_transformer.mblt`: MBLT 형식의 컴파일된 모델

## Stage 3: MXQ 컴파일 (고급 양자화)

MXQ(Mobilint eXeQutable) 형식은 고급 양자화 기법을 적용하고 NPU에서 배포할 수 있도록 모델을 준비합니다.

### Step 3.1: 언어 모델을 MXQ로 컴파일

언어 모델을 MBLT에서 MXQ 형식으로 컴파일합니다. `--target-device` 는 필수입니다 (`aries-rb` 또는 `regulus-rb`):

```bash
# ARIES
python mxq_compile_language.py --target-device aries-rb

# REGULUS (2026-06 이후 고객)
python mxq_compile_language.py --target-device regulus-rb
```

**이 작업의 내용:**

- MBLT 파일 로드: `./mblt/Qwen3-VL-4B-Instruct_text_model.mblt`
- 캘리브레이션 데이터 로드: `./calibration_data/language/npy_files.json`
- 등가 변환을 사용한 고급 양자화 적용
- 임베딩 및 deepstack 입력에 대해 16비트 활성화 구성: `inputs_embeds/reshape`, `deepstack_visual_embeds_0`
- NPU 추론 스키마: `--target-device` 에 따라 자동 설정 (ARIES `all`, REGULUS `single`)
- **회전 행렬 생성** 위치: `./spinWeight/Qwen3-VL-4B-Instruct_text_model/R1/global_rotation.pth`
  - 이 회전 행렬은 **비전 인코더 MXQ 컴파일에 필요합니다**

**주요 구성 (벤치마크 최적 4B 디코더):**

- 캘리브레이션: 모드 0 (Max), 출력 0
- 가중치: 4비트 (query/key/output/ffn/head), value projection은 8비트로 승격, 컴파일 중 float32 weight dtype
- 활성화 16비트 레이어: `["inputs_embeds/reshape", "deepstack_visual_embeds_0"]`
- 추론 스키마: ARIES `all`, REGULUS `single`
- 등가 변환: UD (smoothing_factor=0.8), VO, SpinR1, SpinR2, optimize_ffn (QK 비활성)
- OPTQ 활성화 (act_order, block_size=128, perc_damp=0.01)
- query/key/value/out/ffn 에 대한 weight-scale 탐색 활성화

**출력 파일:**

- `./mxq/Qwen3-VL-4B-Instruct_text_model.mxq`: 배포 준비가 된 양자화된 모델
- `./spinWeight/Qwen3-VL-4B-Instruct_text_model/R1/global_rotation.pth`: 전역 회전 행렬 (비전 인코더에 필요)

### Step 3.2: 비전 인코더를 MXQ로 컴파일

**중요:** 비전 인코더 컴파일에는 언어 모델 컴파일 중 생성된 회전 행렬이 필요하므로 먼저 Step 3.1(언어 모델 MXQ 컴파일)을 완료해야 합니다.

비전 인코더를 MBLT에서 MXQ 형식으로 컴파일합니다. `--target-device` 는 필수입니다 (`aries-rb` 또는 `regulus-rb`):

```bash
# ARIES
python mxq_compile_vision.py --target-device aries-rb

# REGULUS (2026-06 이후 고객)
python mxq_compile_vision.py --target-device regulus-rb
```

**이 작업의 내용:**

- MBLT 파일 로드: `./mblt/Qwen3-VL-4B-Instruct_vision_transformer.mblt`
- 캘리브레이션 데이터 로드: `./calibration_data/vision/npy_files.txt`
- **회전 행렬 로드** 위치: `./spinWeight/Qwen3-VL-4B-Instruct_text_model/R1/global_rotation.pth`
  - 이 행렬은 언어 모델 MXQ 컴파일 중에 생성되었습니다
  - 비전 및 언어 컴포넌트 간의 일관된 양자화를 보장합니다
- 등가 변환을 사용한 고급 양자화 적용:
  - **헤드 출력 채널 회전**: 공용 회전 행렬을 사용하여 비전 인코더 출력을 언어 모델 입력과 정렬
- merger 및 deepstack merger `fc2` 레이어에 대해 16비트 활성화 구성
- 비전 인코더에 대해 다중 코어 컴파일 사용

**주요 구성 (벤치마크 최적 4B 인코더):**

- 캘리브레이션: 모드 1 (MaxPercentile), 출력 0
- 활성화 16비트 레이어: `["model_merger_linear_fc2", "model_deepstack_merger_list_0_linear_fc2", "model_deepstack_merger_list_1_linear_fc2", "model_deepstack_merger_list_2_linear_fc2"]`
- 추론 스키마: ARIES `all`, REGULUS `single`
- 등가 변환: QK, UD, VO, head_out_ch_rotation (언어 모델 회전 행렬 사용), SpinR2, optimize_ffn (SpinR1 비활성)
- 회전 행렬 경로: `./spinWeight/Qwen3-VL-4B-Instruct_text_model/R1/global_rotation.pth`

**회전 행렬이 필요한 이유:**
비전 인코더의 출력은 언어 모델의 입력 공간과 올바르게 정렬되어야 합니다. 언어 모델 양자화 중 생성된 회전 행렬은 비전 특징과 텍스트 임베딩이 동일한 양자화된 공간에 존재하도록 보장하여, 추론 중 비전 및 언어 컴포넌트가 결합될 때 정확도를 유지합니다.

**출력 파일:**

- `./mxq/Qwen3-VL-4B-Instruct_vision_transformer.mxq`: ARIES 배포 준비가 된 양자화된 모델

### 대상 디바이스 (`--target-device`)

언어 모델과 비전 모델의 MXQ 컴파일 스크립트는 모두 `--target-device`로 대상 NPU를 지정합니다. REGULUS는 `inference_scheme="single"`만 지원하므로 `regulus` 디바이스를 지정하면 자동으로 적용됩니다. ARIES 흐름과 마찬가지로, 비전 인코더를 컴파일하기 전에 언어 모델을 먼저 실행해 회전 행렬을 생성해야 합니다.

| 사용자 | `--target-device` |
|---|---|
| ARIES | `aries-rb` |
| REGULUS (2026-06 이후 고객) | `regulus-rb` |

> **참고:** VLM 컴파일은 신형 REGULUS(`regulus-rb`, 2026-06 이후 고객)에서 지원됩니다. 구형 REGULUS(`regulus-ra`, 2026-06 이전 고객)는 이 워크플로를 지원하지 않습니다.

출력은 대상 디바이스와 관계없이 동일한 `./mxq/` 경로에 저장됩니다.

### Step 3.3: 회전된 토큰 임베딩 준비

두 모델을 모두 MXQ 형식으로 컴파일한 후 추론을 위한 회전된 토큰 임베딩 가중치를 준비해야 합니다.

**중요:** 이 단계는 언어 모델 컴파일의 회전 행렬이 필요하므로 두 MXQ 컴파일(Step 3.1 및 3.2)을 모두 완료한 후에 수행해야 합니다.

#### 회전된 토큰 임베딩 가중치 준비

토큰 임베딩 가중치(`model.language_model.embed_tokens.weight`)를 다운로드하고 회전을 적용합니다:

```bash
python get_safetensors.py
```

**이 작업의 내용:**

- HuggingFace에서 `model.safetensors` 다운로드 (Qwen3-VL-4B는 샤딩되지 않은 단일 safetensors 파일로 배포됨)
- 토큰 임베딩 가중치(`model.language_model.embed_tokens.weight`) 추출 — 토큰 ID를 hidden state 벡터로 매핑하는 룩업 테이블
- 언어 모델 MXQ 컴파일의 회전 행렬 적용:
  - 회전 행렬 로드 위치: `./spinWeight/Qwen3-VL-4B-Instruct_text_model/R1/global_rotation.pth`
  - 토큰 임베딩 가중치에 회전 행렬을 우측 곱(`W @ R1`)하여 양자화된 공간과 정렬
- 결과를 `./mxq/model.safetensors`에 저장

**토큰 임베딩 회전이 필요한 이유:**
MXQ 컴파일 시 `SpinR1` 등가 변환이 언어 모델 내부 가중치를 회전된 공간으로 변환합니다. 그러나 토큰 임베딩 레이어는 MXQ 컴파일 대상이 아닙니다 — 추론 시 CPU에서 룩업으로 실행됩니다. 따라서 토큰 임베딩 가중치를 동일한 회전 행렬로 사전 회전하여, 룩업 결과가 양자화된 모델의 입력 공간과 일치하도록 해야 합니다.

**참고:** 출력 파일명 `model.safetensors`는 HuggingFace `PreTrainedModel.from_pretrained()` 규약에 의해 고정됩니다. 이름과 달리 이 파일에는 **회전된 토큰 임베딩 가중치만** 포함되어 있습니다.

**출력 파일:**

- `./mxq/model.safetensors`: 회전된 토큰 임베딩 가중치 (`model.language_model.embed_tokens.weight`)

**중요:** 이 스크립트를 실행한 후 `./mxq/` 디렉토리에는 컴파일된 모델 파일이 있습니다:

1. `Qwen3-VL-4B-Instruct_text_model.mxq` (컴파일된 언어 모델)
2. `Qwen3-VL-4B-Instruct_vision_transformer.mxq` (컴파일된 비전 인코더)
3. `model.safetensors` (회전된 토큰 임베딩 가중치)

추가 파일 복사가 필요하지 않습니다!

## 전체 컴파일 파이프라인

전체 VLM을 컴파일하는 명령어 시퀀스는 다음과 같습니다:

```bash
# Stage 1: 캘리브레이션 데이터 생성

# COCO 데이터셋에서 캘리브레이션 이미지 다운로드
python download_images.py

# 캘리브레이션 데이터 생성 (언어 + 비전)
python generate_calibration_data.py \
    --model-name Qwen/Qwen3-VL-4B-Instruct \
    --output-dir ./calibration_data \
    --num-samples 100 \
    --max-new-tokens 500

# Stage 2: MBLT 컴파일

# 언어 모델을 MBLT로 컴파일 (--target-device 필수)
python mblt_compile_language.py --target-device aries-rb

# 비전 인코더를 MBLT로 컴파일
python mblt_compile_vision.py --target-device aries-rb

# Stage 3: MXQ 컴파일 및 추론 준비
# 중요: 언어 모델을 먼저 컴파일해야 합니다 (회전 행렬 생성)
python mxq_compile_language.py --target-device aries-rb

# 그런 다음 비전 인코더 컴파일 (언어 모델의 회전 행렬 사용)
python mxq_compile_vision.py --target-device aries-rb

# 회전된 토큰 임베딩 준비
python get_safetensors.py

# 모든 필요한 파일이 이제 mxq/ 디렉토리에 있습니다:
# - Qwen3-VL-4B-Instruct_text_model.mxq
# - Qwen3-VL-4B-Instruct_vision_transformer.mxq
# - model.safetensors
```

## 컴파일 흐름 이해

### 언어 모델 파이프라인

```text
[이미지 다운로드] -> images/*.jpg (100개의 COCO 이미지)
    |
원본 모델 (HF) + 캘리브레이션 이미지
    |
[캘리브레이션] -> calibration_data/language/*.npy
    |
[MBLT 컴파일] -> Qwen3-VL-4B-Instruct_text_model.mblt
    |
[MXQ 컴파일] -> Qwen3-VL-4B-Instruct_text_model.mxq
    |
    +-> global_rotation.pth (비전 인코더에 필요)
```

### 비전 인코더 파이프라인

```text
[이미지 다운로드] -> images/*.jpg (100개의 COCO 이미지)
    |
원본 모델 (HF) + 캘리브레이션 이미지
    |
[캘리브레이션] -> calibration_data/vision/*.npy
    |
[MBLT 컴파일] -> Qwen3-VL-4B-Instruct_vision_transformer.mblt
    |
[MXQ 컴파일] -> Qwen3-VL-4B-Instruct_vision_transformer.mxq
    |            (요구사항: 언어 모델의 global_rotation.pth)
```

### 토큰 임베딩 준비

```text
[get_safetensors.py] -> model.safetensors
                        (회전된 토큰 임베딩 가중치)
```

### 주요 종속성

1. 비전 인코더 MXQ 컴파일은 언어 모델 MXQ 컴파일의 회전 행렬이 **필수**입니다
2. 항상 `mxq_compile_vision.py` **이전에** `mxq_compile_language.py`를 실행하세요
3. 두 MBLT 파일은 독립적으로 컴파일할 수 있지만, MXQ 파일은 위 순서를 따라야 합니다
4. `get_safetensors.py`는 언어 모델 MXQ 컴파일의 회전 행렬이 필요합니다
5. 모든 출력 파일(2개의 MXQ 모델, model.safetensors)이 모두 같은 디렉토리에 있어야 합니다

## 출력 요약

모든 단계를 완료한 후 다음을 갖게 됩니다:

### 캘리브레이션 데이터

- `calibration_data/language/`: 메타데이터가 포함된 언어 모델 캘리브레이션 샘플
- `calibration_data/vision/`: 메타데이터가 포함된 비전 인코더 캘리브레이션 샘플

### MBLT 모델 (하드웨어 독립적) - `mblt/`에 위치

- `Qwen3-VL-4B-Instruct_text_model.mblt`: MBLT 형식의 언어 모델
- `Qwen3-VL-4B-Instruct_vision_transformer.mblt`: MBLT 형식의 비전 인코더

### MXQ 모델 및 컴파일 산출물 - `mxq/`에 위치

컴파일 산출물이 이 단일 디렉토리에 정리됩니다. 런타임 단계에서 HF 저장소에서 clone한 `config.json`, proxy 클래스, tokenizer/processor와 결합해 배포 가능한 모델 폴더를 만듭니다:

- `Qwen3-VL-4B-Instruct_text_model.mxq`: 양자화된 언어 모델
- `Qwen3-VL-4B-Instruct_vision_transformer.mxq`: 양자화된 비전 인코더
- `model.safetensors`: 회전된 토큰 임베딩 가중치 (`model.language_model.embed_tokens.weight`)

## 문제 해결

### 메모리 부족 (OOM) 오류

- 캘리브레이션 스크립트에서 `--num-samples` 감소
- 언어 캘리브레이션에서 `--max-new-tokens` 감소
- 다른 GPU 집약적 애플리케이션 종료

### 회전 행렬 누락 오류

비전 인코더 MXQ 컴파일이 회전 행렬 누락 오류로 실패하는 경우:

```bash
FileNotFoundError: ./spinWeight/Qwen3-VL-4B-Instruct_text_model/R1/global_rotation.pth
```

**해결 방법:** 먼저 `mxq_compile_language.py`를 실행하여 회전 행렬을 생성하세요.

### 캘리브레이션 데이터를 찾을 수 없음

MXQ 컴파일 스크립트의 캘리브레이션 데이터 경로가 실제 캘리브레이션 데이터 위치와 일치하는지 확인하세요:

- 언어: `./calibration_data/language/npy_files.json`
- 비전: 데이터가 다른 위치에 있는 경우 `mxq_compile_vision.py`의 경로를 업데이트하세요

### 모델 다운로드 문제

- HuggingFace에서 모델 동의를 수락했는지 확인
- 액세스 토큰이 유효한지 확인: `huggingface-cli whoami`
- 인터넷 연결 및 HuggingFace 상태 확인

### 이미지를 찾을 수 없음

```bash
FileNotFoundError: No images found in images/ directory
```

**해결 방법:** 이미지 다운로드 스크립트를 실행하세요:

```bash
python download_images.py
```

이렇게 하면 COCO 데이터셋에서 100개의 이미지를 `images/` 디렉토리에 다운로드합니다.

## 배포

모든 컴파일 단계를 완료한 후 `./mxq/` 디렉토리에는 런타임 모델 폴더를 만들기 위한 컴파일 산출물이 포함됩니다:

1. **Qwen3-VL-4B-Instruct_text_model.mxq** - 컴파일된 언어 모델
2. **Qwen3-VL-4B-Instruct_vision_transformer.mxq** - 컴파일된 비전 인코더
3. **model.safetensors** - 회전된 토큰 임베딩 가중치 (`model.language_model.embed_tokens.weight`)

이 산출물만으로는 배포할 수 없습니다. 런타임 단계(`prepare_model.py`)에서 Hugging Face 저장소를 clone하여 `config.json`, proxy 클래스, tokenizer/processor를 가져온 뒤 이 컴파일 산출물을 넣어 자체 완결형(self-contained) 모델 폴더를 만듭니다. 자세한 내용은 런타임 추론 튜토리얼을 참조하세요.

## 다음 단계: 추론 실행

컴파일된 모델로 추론을 실행하려면 [런타임 추론 튜토리얼](../../runtime/python/vlm/README.KR.md)을 참조하세요.

런타임 튜토리얼에서는 다음 방법을 보여줍니다:

- mblt-model-zoo를 사용하여 컴파일된 MXQ 모델 로드
- 이미지-텍스트-텍스트 추론 실행
- 프롬프트 및 생성 매개변수 사용자 정의
- 다중 턴 대화 처리
- 여러 이미지 처리

## 참고 자료

- [Qwen3-VL 모델 카드](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [Mobilint 문서](https://docs.mobilint.com)

## 지원

문제나 질문이 있는 경우:

- 위의 문제 해결 섹션을 확인하세요
- qbcompiler SDK 문서를 검토하세요
- 상세한 오류 로그와 함께 Mobilint 지원팀에 문의하세요

---

**참고:** 이 튜토리얼은 VLM 컴파일의 전체 파이프라인을 보여줍니다. 여기에 표시된 기법은 모델 로딩 및 패칭 코드를 적절히 수정하여 다른 비전-언어 모델에 적용할 수 있습니다.
