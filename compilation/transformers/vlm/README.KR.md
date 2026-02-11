# 비전 언어 모델 (VLM) 컴파일

이 튜토리얼은 Mobilint qbcompiler 컴파일러를 사용하여 비전 언어 모델(VLM)을 컴파일하는 방법에 대한 상세한 지침을 제공합니다.

이 튜토리얼에서는 Qwen에서 개발한 최첨단 비전-언어 모델인 [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) 모델을 사용합니다.

## 개요

VLM 컴파일 과정은 세 가지 주요 단계로 구성됩니다:

1. **캘리브레이션 데이터 생성**: 양자화를 위한 캘리브레이션 데이터셋 생성
2. **MBLT 컴파일**: 모델을 MBLT(Mobilint Binary LayouT) 형식으로 컴파일
3. **MXQ 컴파일**: 고급 양자화를 적용하고 배포를 위해 `.mxq` 형식으로 컴파일

컴파일 과정은 **언어 모델**(디코더)과 **비전 인코더** 컴포넌트에 대해 별도로 수행됩니다.

컴파일 후에는 Aries 2 하드웨어에서 배포할 준비가 된 모든 필요한 파일이 `mxq/` 디렉토리에 생성됩니다.

## 사전 요구사항

시작하기 전에 다음이 있는지 확인하세요:

- Python 3.8 이상
- qbcompiler SDK 컴파일러 설치 (버전 >= 1.0.1 필요)
- (선택사항) 캘리브레이션 및 컴파일을 위한 CUDA 지원 GPU
- 충분한 디스크 공간 (모델 + 캘리브레이션 데이터용 약 20GB)

### 필수 의존성 패키지 설치

컴파일에 필요한 Python 패키지를 설치해야 합니다:

```bash
pip install transformers torch torchvision qwen-vl-utils datasets
```

### 캘리브레이션 이미지 다운로드

캘리브레이션 과정은 COCO 데이터셋의 이미지를 사용합니다. 100개의 이미지를 자동으로 가져오는 다운로드 스크립트가 제공됩니다:

```bash
cd calibration
python download_images.py
```

**이 작업의 내용:**

- HuggingFace datasets를 사용하여 COCO 2017 validation에서 100개의 이미지 다운로드
- 이미지를 224x224 해상도로 자동 리사이즈
- 이미지를 JPEG 파일로 `images/` 디렉토리에 저장
- COCO 다운로드가 실패하면 대체할 합성 샘플 이미지 생성합니다.

**출력:**

- `images/image_0000.jpg`부터 `images/image_0099.jpg`까지

캘리브레이션 스크립트는 `images/` 디렉토리의 모든 이미지를 자동으로 사용하고 다양한 프롬프트(상세 설명, 시각적 추론, 카운팅, 공간 이해 등)를 순환하여 캘리브레이션 다양성을 보장합니다.

## Stage 1: 캘리브레이션 데이터 생성

캘리브레이션 데이터는 양자화에 필수적이며, 컴파일러가 모델의 일반적인 활성화 범위를 이해하는 데 도움이 됩니다.

### Step 1.1: 언어 모델 캘리브레이션 데이터 생성

언어 모델(디코더)에 대한 캘리브레이션 데이터를 생성합니다:

```bash
cd calibration
python generate_language_calibration_data.py \
    --model-name Qwen/Qwen2-VL-2B-Instruct \
    --output-dir ./calibration_data/language \
    --num-samples 100 \
    --max-new-tokens 500
```

**매개변수:**

- `--model-name`: HuggingFace 모델 식별자
- `--output-dir`: 캘리브레이션 데이터를 저장할 디렉토리
- `--num-samples`: 캘리브레이션 샘플 수 (기본값: 사용 가능한 모든 이미지)
- `--max-new-tokens`: 샘플당 생성할 최대 토큰 수 (더 긴 시퀀스 캡처)

**이 작업의 내용:**

- `images/` 폴더에서 모든 이미지 로드 (이전에 다운로드한 100개의 JPEG 이미지)
- 20가지 다양한 프롬프트 유형 순환 (객체 식별, 상세 설명, 시각적 추론, 공간 이해 등)
- 비전 특징이 텍스트 임베딩에 병합된 후 `inputs_embeds` 텐서 캡처
- 메타데이터와 함께 캘리브레이션 데이터를 `.npy` 파일로 저장

**출력 구조:**

```text
calibration_data/language/
 sample_000/
    inputs_embeds.npy    # [1, seq_len, 1536]
 sample_001/
    inputs_embeds.npy
 ...
 metadata.json            # 샘플 정보
 npy_files.txt           # 모든 .npy 경로 목록
```

### Step 1.2: 비전 인코더 캘리브레이션 데이터 생성

비전 인코더에 대한 캘리브레이션 데이터를 생성합니다:

```bash
python generate_vision_calibration_data.py \
    --model-name Qwen/Qwen2-VL-2B-Instruct \
    --output-dir ./calibration_data/vision \
    --num-samples 100
```

**매개변수:**

- `--model-name`: HuggingFace 모델 식별자
- `--output-dir`: 캘리브레이션 데이터를 저장할 디렉토리
- `--num-samples`: 캘리브레이션 샘플 수 (기본값: 사용 가능한 모든 이미지)

**참고:** 비전 인코더 캘리브레이션의 경우 이미지 크기는 224x224로 고정됩니다.

**이 작업의 내용:**

- `images/` 폴더에서 모든 이미지 로드 (이전에 다운로드한 100개의 JPEG 이미지)
- 다양한 프롬프트 순환 (언어 캘리브레이션과 동일)
- 비전 인코더 입력(픽셀 값) 캡처
- Aries 2 아키텍처와 호환되는 형식으로 리셰이프: `[896, 56, 6]`
- 메타데이터와 함께 캘리브레이션 데이터를 `.npy` 파일로 저장

**출력 구조:**

```text
calibration_data/vision/
 sample_000/
    images.npy          # [896, 56, 6]
 sample_001/
    images.npy
 ...
 metadata.json           # 샘플 정보
 npy_files.txt          # 모든 .npy 경로 목록
```

## Stage 2: MBLT 컴파일

MBLT(Mobilint Binary LayouT)는 모델 그래프와 가중치를 하드웨어 독립적인 방식으로 나타내는 중간 형식입니다.

### Step 2.1: 언어 모델을 MBLT로 컴파일

언어 모델(디코더)을 MBLT 형식으로 컴파일합니다:

```bash
cd ../compile
python mblt_compile_language.py
```

**이 작업의 내용:**

- 샘플 생성 중 언어 모델 입력 캡처
- 시퀀스 길이 차원을 동적으로 표시 (가변 길이 입력용)
- Aries 2 호환 아키텍처 패치 적용:
  - **사전 캐시된 RoPE 임베딩**: 런타임 삼각 함수 연산 제거
  - **마지막 쿼리 슬라이싱**: 디코드 단계를 위해 최종 디코더 레이어 최적화
  - **상태 저장 KV 캐시 래퍼**: 효율적인 자기 회귀 생성 활성화
  - **동적 형태 처리**: 가변 시퀀스 길이 지원
- PyTorch FX 추적을 사용하여 qbcompiler의 ModelParser로 모델 컴파일
- 어텐션 연산자에 대한 동적 형태 구성
- MBLT 바이너리 형식으로 직렬화
- 원본 모델과 비교하여 출력 검증

**주요 변환:**

- 입력 임베딩 차원을 동적으로 표시: `[batch, seq_len, hidden_size]`
- 가변 시퀀스를 위해 어텐션 마스크 및 위치 ID를 동적으로 표시
- 자기 회귀 생성을 위해 캐시 위치를 동적으로 표시
- 최대 시퀀스 길이(16384)에 대해 RoPE 임베딩 사전 계산

**출력 파일:**

- `./mblt/Qwen2-VL-2B-Instruct_text_model.mblt`: MBLT 형식의 컴파일된 모델
- `./mblt/Qwen2-VL-2B-Instruct_text_model.infer`: 검증을 위한 추론 값
- `./mblt/Qwen2-VL-2B-Instruct_text_model.json`: 원본 모델과의 비교 결과

### Step 2.2: 비전 인코더를 MBLT로 컴파일

비전 인코더를 MBLT 형식으로 컴파일합니다:

```bash
python mblt_compile_vision.py
```

**이 작업의 내용:**

- 샘플 추론 중 비전 인코더 입력 캡처
- 픽셀 값을 Aries 2 호환 형식으로 재처리
- Aries 2 호환 아키텍처 패치 적용:
  - **3D2D 컨볼루션**: NPU 최적화를 위해 3D 컨볼루션을 2D로 변환
  - **분할 QKV 프로젝션**: 더 나은 병렬화를 위해 Query, Key, Value 프로젝션 분리
  - **사전 계산된 RoPE 임베딩**: 런타임 삼각 함수 연산 제거
  - **병합된 패치화 연산**: 메모리 전송 감소
- PyTorch FX 추적을 사용하여 qbcompiler의 ModelParser로 모델 컴파일
- 모든 입력 상수에 대해 데이터 형식을 NHWC(Aries 2 친화적)로 설정
- MBLT 바이너리 형식으로 직렬화
- 원본 모델과 비교하여 출력 검증

**주요 변환:**

- 픽셀 값을 HuggingFace 형식 `[num_patches, channels*patch_size^2]`에서 Aries 2 형식 `[batch, channels*temporal, height, width]`로 재처리
- 3D 시간 컨볼루션을 2D 공간 컨볼루션으로 변환
- 병렬 실행을 위해 QKV 어텐션 프로젝션 분리
- 이미지 그리드 차원을 기반으로 RoPE 임베딩 사전 계산

**출력 파일:**

- `./mblt/Qwen2-VL-2B-Instruct_vision_transformer.mblt`: MBLT 형식의 컴파일된 모델
- `./mblt/Qwen2-VL-2B-Instruct_vision_transformer.infer`: 검증을 위한 추론 값
- `./mblt/Qwen2-VL-2B-Instruct_vision_transformer.json`: 원본 모델과의 비교 결과

## Stage 3: MXQ 컴파일 (고급 양자화)

MXQ(Mobilint Quantized) 형식은 고급 양자화 기법을 적용하고 Aries 2 하드웨어에서 배포할 수 있도록 모델을 준비합니다.

### Step 3.1: 언어 모델을 MXQ로 컴파일

언어 모델을 MBLT에서 MXQ 형식으로 컴파일합니다:

```bash
python mxq_compile_language.py
```

**이 작업의 내용:**

- MBLT 파일 로드: `./mblt/Qwen2-VL-2B-Instruct_text_model.mblt`
- 캘리브레이션 데이터 로드: `../calibration/calibration_data/language/npy_files.txt`
- 등가 변환을 사용한 고급 양자화 적용
- 입력 임베딩에 대해 16비트 활성화 구성: `inputs_embeds/reshape`
- 언어 모델에 대해 단일 코어 컴파일 사용
- LLM 특화 최적화 활성화
- **회전 행렬 생성** 위치: `/tmp/qbcompiler/spinWeight/Qwen2-VL-2B-Instruct_text_model/R1/global_rotation.pth`
  - 이 회전 행렬은 **비전 인코더 MXQ 컴파일에 필요합니다**

**주요 구성:**

- 캘리브레이션 모드: 1 (표준 캘리브레이션)
- 활성화 16비트 레이어: `["inputs_embeds/reshape"]`
- 추론 스키마: `single` (단일 코어 실행)
- 단일 코어 컴파일: `True` (언어 모델에 최적화)
- 등가 변환: QK, UD (학습 포함), SPIN R1, SPIN R2

**출력 파일:**

- `./mxq/Qwen2-VL-2B-Instruct_text_model.mxq`: Aries 2 배포 준비가 된 양자화된 모델
- `/tmp/qbcompiler/spinWeight/Qwen2-VL-2B-Instruct_text_model/R1/global_rotation.pth`: 전역 회전 행렬 (비전 인코더에 필요)

### Step 3.2: 비전 인코더를 MXQ로 컴파일

**중요:** 비전 인코더 컴파일에는 언어 모델 컴파일 중 생성된 회전 행렬이 필요하므로 먼저 Step 3.1(언어 모델 MXQ 컴파일)을 완료해야 합니다.

비전 인코더를 MBLT에서 MXQ 형식으로 컴파일합니다:

```bash
python mxq_compile_vision.py
```

**이 작업의 내용:**

- MBLT 파일 로드: `./mblt/Qwen2-VL-2B-Instruct_vision_transformer.mblt`
- 캘리브레이션 데이터 로드: `../calibration/calibration_data/vision/npy_files.txt`
- **회전 행렬 로드** 위치: `/tmp/qbcompiler/spinWeight/Qwen2-VL-2B-Instruct_text_model/R1/global_rotation.pth`
  - 이 행렬은 언어 모델 MXQ 컴파일 중에 생성되었습니다
  - 비전 및 언어 컴포넌트 간의 일관된 양자화를 보장합니다
- 등가 변환을 사용한 고급 양자화 적용:
  - **헤드 출력 채널 회전**: 공용 회전 행렬을 사용하여 비전 인코더 출력을 언어 모델 입력과 정렬
- 병합 레이어에 대해 16비트 활성화 구성: `model_merger_fc2` (향후 필요하지 않을 수 있습니다)
- 비전 인코더에 대해 다중 코어 컴파일 사용

**주요 구성:**

- 캘리브레이션 출력 모드: 1 (표준 출력 캘리브레이션)
- 활성화 16비트 레이어: `["model_merger_fc2"]`
- 추론 스키마: `multi` (다중 코어 실행)
- 등가 변환: 헤드 출력 채널 회전 (언어 모델 회전 행렬 사용)
- 회전 행렬 경로: `/tmp/qbcompiler/spinWeight/Qwen2-VL-2B-Instruct_text_model/R1/global_rotation.pth`

**회전 행렬이 필요한 이유:**
비전 인코더의 출력은 언어 모델의 입력 공간과 올바르게 정렬되어야 합니다. 언어 모델 양자화 중 생성된 회전 행렬은 비전 특징과 텍스트 임베딩이 동일한 양자화된 공간에 존재하도록 보장하여, 추론 중 비전 및 언어 컴포넌트가 결합될 때 정확도를 유지합니다.

**출력 파일:**

- `./mxq/Qwen2-VL-2B-Instruct_vision_transformer.mxq`: Aries 2 배포 준비가 된 양자화된 모델

### Step 3.3: 추론 구성 파일 준비

두 모델을 모두 MXQ 형식으로 컴파일한 후 추론을 위한 구성 파일을 준비해야 합니다. 이 단계는 필요한 모델 구성 파일을 다운로드하고 컴파일된 MXQ 모델과 함께 사용할 수 있도록 준비합니다.

**중요:** 이 단계는 언어 모델 컴파일의 회전 행렬이 필요하므로 두 MXQ 컴파일(Step 3.1 및 3.2)을 모두 완료한 후에 수행해야 합니다.

#### 모델 구성 가져오기

먼저 모델 구성 파일을 다운로드하고 준비합니다:

```bash
python get_config.py
```

**이 작업의 내용:**

- HuggingFace 모델 저장소에서 `config.json` 다운로드
- 추론 파일을 위한 `./mxq/` 디렉토리 생성
- 컴파일된 MXQ 모델 파일을 가리키도록 구성 수정:
  - `mxq_path`를 `"Qwen2-VL-2B-Instruct_text_model.mxq"`로 설정
  - `vision_config.mxq_path`를 `"Qwen2-VL-2B-Instruct_vision_transformer.mxq"`로 설정
- 모델 아키텍처 설정 업데이트:
  - `architectures`를 `["MobilintQwen2VLForConditionalGeneration"]`로 변경
  - `model_type`을 `'mobilint-qwen2_vl'`로 변경
  - `max_position_embeddings`를 32768로 설정
  - `sliding_window`를 32768로 설정
  - `tie_word_embeddings` 활성화
- 수정된 구성을 `./mxq/config.json`에 저장

#### 모델 임베딩 가져오기

다음으로 적절한 회전을 적용하여 임베딩 가중치를 다운로드하고 준비합니다:

```bash
python get_safetensors.py
```

**이 작업의 내용:**

- HuggingFace에서 `model-00001-of-00002.safetensors` 다운로드 (임베딩 가중치 포함)
- `model.embed_tokens.weight` 텐서 추출
- 언어 모델 MXQ 컴파일의 회전 행렬 적용:
  - 회전 행렬 로드 위치: `/tmp/qbcompiler/spinWeight/Qwen2-VL-2B-Instruct_text_model/R1/global_rotation.pth`
  - 임베딩 텐서에 회전 행렬을 곱하여 양자화된 공간과 정렬
- 회전된 임베딩 텐서를 `./mxq/model.safetensors`에 저장

**임베딩 회전이 필요한 이유:**
임베딩 레이어는 언어 모델 양자화 중에 사용된 것과 동일한 회전 행렬로 회전되어야 합니다. 이는 입력 임베딩이 언어 모델의 나머지 부분과 동일한 양자화된 공간에 있도록 보장하여 추론 파이프라인 전체에서 정확도와 일관성을 유지합니다.

**출력 파일:**

- `./mxq/config.json`: MXQ 파일을 가리키는 수정된 모델 구성
- `./mxq/model.safetensors`: 양자화된 모델과 정렬된 회전된 임베딩 가중치

**중요:** 이 스크립트를 실행한 후 `./mxq/` 디렉토리에 추론에 필요한 4개의 파일이 모두 생성됩니다:

1. `Qwen2-VL-2B-Instruct_text_model.mxq` (컴파일된 언어 모델)
2. `Qwen2-VL-2B-Instruct_vision_transformer.mxq` (컴파일된 비전 인코더)
3. `config.json` (모델 구성)
4. `model.safetensors` (회전된 임베딩)

추가 파일 복사가 필요하지 않습니다!

## 전체 컴파일 파이프라인

전체 VLM을 컴파일하는 명령어 시퀀스는 다음과 같습니다:

```bash
# Stage 1: 캘리브레이션 데이터 생성
cd /workspace/mblt-sdk-tutorial/compilation/transformers/vlm/calibration

# COCO 데이터셋에서 캘리브레이션 이미지 다운로드
python download_images.py

# 언어 캘리브레이션 데이터 생성
python generate_language_calibration_data.py \
    --model-name Qwen/Qwen2-VL-2B-Instruct \
    --output-dir ./calibration_data/language \
    --num-samples 100 \
    --max-new-tokens 500

# 비전 캘리브레이션 데이터 생성
python generate_vision_calibration_data.py \
    --model-name Qwen/Qwen2-VL-2B-Instruct \
    --output-dir ./calibration_data/vision \
    --num-samples 100

# Stage 2: MBLT 컴파일
cd ../compile

# 언어 모델을 MBLT로 컴파일
python mblt_compile_language.py

# 비전 인코더를 MBLT로 컴파일
python mblt_compile_vision.py

# Stage 3: MXQ 컴파일 및 추론 준비
# 중요: 언어 모델을 먼저 컴파일해야 합니다 (회전 행렬 생성)
python mxq_compile_language.py

# 그런 다음 비전 인코더 컴파일 (언어 모델의 회전 행렬 사용)
python mxq_compile_vision.py

# 추론 구성 파일 준비 (config.json 및 model.safetensors)
python get_config.py
python get_safetensors.py

# 모든 필요한 파일이 이제 mxq/ 디렉토리에 있습니다:
# - Qwen2-VL-2B-Instruct_text_model.mxq
# - Qwen2-VL-2B-Instruct_vision_transformer.mxq
# - config.json
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
[MBLT 컴파일] -> Qwen2-VL-2B-Instruct_text_model.mblt
    |
[MXQ 컴파일] -> Qwen2-VL-2B-Instruct_text_model.mxq
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
[MBLT 컴파일] -> Qwen2-VL-2B-Instruct_vision_transformer.mblt
    |
[MXQ 컴파일] -> Qwen2-VL-2B-Instruct_vision_transformer.mxq
    |            (요구사항: 언어 모델의 global_rotation.pth)
```

### 구성 파일 준비

```text
[get_config.py] -> config.json
                   (MXQ 경로로 수정됨)

[get_safetensors.py] -> model.safetensors
                        (회전이 적용된 임베딩 가중치)
```

### 주요 종속성

1. 비전 인코더 MXQ 컴파일은 언어 모델 MXQ 컴파일의 회전 행렬이 **필수**입니다
2. 항상 `mxq_compile_vision.py` **이전에** `mxq_compile_language.py`를 실행하세요
3. 두 MBLT 파일은 독립적으로 컴파일할 수 있지만, MXQ 파일은 위 순서를 따라야 합니다
4. `get_safetensors.py`는 언어 모델 MXQ 컴파일의 회전 행렬이 필요합니다
5. 배포를 위해 4개의 출력 파일(2개의 MXQ 모델, config.json, model.safetensors)이 모두 같은 디렉토리에 있어야 합니다

## 출력 요약

모든 단계를 완료한 후 다음을 갖게 됩니다:

### 캘리브레이션 데이터

- `calibration_data/language/`: 메타데이터가 포함된 언어 모델 캘리브레이션 샘플
- `calibration_data/vision/`: 메타데이터가 포함된 비전 인코더 캘리브레이션 샘플

### MBLT 모델 (하드웨어 독립적) - `compile/mblt/`에 위치

- `Qwen2-VL-2B-Instruct_text_model.mblt`: MBLT 형식의 언어 모델
- `Qwen2-VL-2B-Instruct_vision_transformer.mblt`: MBLT 형식의 비전 인코더

### MXQ 모델 및 배포 파일 - `compile/mxq/`에 위치

배포에 필요한 모든 파일이 이 단일 디렉토리에 있습니다:

- `Qwen2-VL-2B-Instruct_text_model.mxq`: 양자화된 언어 모델
- `Qwen2-VL-2B-Instruct_vision_transformer.mxq`: 양자화된 비전 인코더
- `config.json`: MXQ 경로가 포함된 모델 구성
- `model.safetensors`: 회전된 임베딩 가중치

### 검증 파일 - `compile/`에 위치

- `*.infer`: 검증을 위한 추론 값
- `*.json`: 원본 모델과의 비교 결과

## 문제 해결

### 메모리 부족 (OOM) 오류

- 캘리브레이션 스크립트에서 `--num-samples` 감소
- 언어 캘리브레이션에서 `--max-new-tokens` 감소
- 다른 GPU 집약적 애플리케이션 종료

### 회전 행렬 누락 오류

비전 인코더 MXQ 컴파일이 회전 행렬 누락 오류로 실패하는 경우:

```bash
FileNotFoundError: /tmp/qbcompiler/spinWeight/qwen2vl_language/R1/global_rotation.pth
```

**해결 방법:** 먼저 `mxq_compile_language.py`를 실행하여 회전 행렬을 생성하세요.

### 캘리브레이션 데이터를 찾을 수 없음

MXQ 컴파일 스크립트의 캘리브레이션 데이터 경로가 실제 캘리브레이션 데이터 위치와 일치하는지 확인하세요:

- 언어: `../calibration/calibration_data/language/npy_files.txt`
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
cd calibration
python download_images.py
```

이렇게 하면 COCO 데이터셋에서 100개의 이미지를 `images/` 디렉토리에 다운로드합니다.

## 배포

모든 컴파일 단계를 완료한 후 `./mxq/` 디렉토리에는 배포에 필요한 4개의 파일이 모두 포함됩니다:

1. **Qwen2-VL-2B-Instruct_text_model.mxq** - 컴파일된 언어 모델
2. **Qwen2-VL-2B-Instruct_vision_transformer.mxq** - 컴파일된 비전 인코더
3. **config.json** - MXQ 경로가 포함된 모델 구성
4. **model.safetensors** - 회전된 임베딩 가중치

이 파일들은 Mobilint 런타임을 사용하여 Aries 2 하드웨어에서 배포할 준비가 되었습니다.

## 다음 단계: 추론 실행

컴파일된 모델로 추론을 실행하려면 [런타임 추론 튜토리얼](../../../runtime/transformers/vlm/README.md)을 참조하세요.

런타임 튜토리얼에서는 다음 방법을 보여줍니다:

- mblt-model-zoo를 사용하여 컴파일된 MXQ 모델 로드
- 이미지-텍스트-텍스트 추론 실행
- 프롬프트 및 생성 매개변수 사용자 정의
- 다중 턴 대화 처리
- 여러 이미지 처리

## 참고 자료

- [Qwen2-VL 모델 카드](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [Mobilint 문서](https://docs.mobilint.com)

## 지원

문제나 질문이 있는 경우:

- 위의 문제 해결 섹션을 확인하세요
- qbcompiler SDK 문서를 검토하세요
- 상세한 오류 로그와 함께 Mobilint 지원팀에 문의하세요

---

**참고:** 이 튜토리얼은 VLM 컴파일의 전체 파이프라인을 보여줍니다. 여기에 표시된 기법은 모델 로딩 및 패칭 코드를 적절히 수정하여 다른 비전-언어 모델에 적용할 수 있습니다.
