# LLM(대규모 언어 모델) 컴파일

이 튜토리얼은 Mobilint `qbcompiler`로 대규모 언어 모델(LLM)을 컴파일하는 방법을 설명합니다.

이 튜토리얼에서는 Meta가 개발한 1B 파라미터 언어 모델 [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)를 사용합니다.

## 개요

컴파일 과정은 세 가지 주요 단계로 구성됩니다:

1. **모델 준비**: 모델을 다운로드하고 임베딩 가중치를 추출합니다.
2. **캘리브레이션 데이터 생성**: Wikipedia 기사에서 캘리브레이션 데이터를 생성합니다.
3. **모델 컴파일**: 8-bit 양자화로 모델을 `.mxq` 형식으로 변환합니다.

## 사전 준비

- qbcompiler SDK (버전 >= 1.0.1)
- (선택) CUDA GPU (컴파일 시간 단축)
- Hugging Face 계정 및 Llama 모델 접근 권한

```bash
pip install -r requirements.txt
```text

## 1단계: 모델 다운로드

[Hugging Face](https://huggingface.co/)에 가입하고 [모델 페이지](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)에서 라이선스에 동의한 후 로그인합니다:

```bash
huggingface-cli login --token <your_huggingface_token>
```text

모델을 다운로드하고 임베딩 가중치를 추출합니다. 임베딩 레이어는 추론 시 CPU에서 실행되며, 나머지 모델은 NPU에서 실행됩니다.

```bash
python download_model.py \
  --repo-id meta-llama/Llama-3.2-1B-Instruct \
  --embedding-path ./embedding.pt
```text

**출력:**

- `embedding.pt` — 임베딩 가중치 행렬 `[vocab_size, embed_dim]`

## Step 2: 캘리브레이션 데이터 생성

[Wikipedia 기사](https://huggingface.co/datasets/wikimedia/wikipedia)에서 캘리브레이션 데이터를 생성합니다. 텍스트를 토큰화하고 임베딩 벡터로 변환하여 양자화 캘리브레이션에 사용합니다.

```bash
python generate_calib.py \
  --model-tag meta-llama/Llama-3.2-1B-Instruct \
  --embedding-path ./embedding.pt \
  --tokenizer-path meta-llama/Llama-3.2-1B-Instruct \
  --output-dir ./calibration_data
```text

**출력:**

- `./calibration_data/datas/meta-llama-Llama-3.2-1B-Instruct/en/` — 128개 캘리브레이션 샘플 (`.npy`)

## Step 3: 모델 컴파일 (8-bit)

8-bit 양자화로 모델을 `.mxq` 형식으로 컴파일합니다.

```bash
python generate_mxq.py \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --calib-data-path ./calibration_data/datas/meta-llama-Llama-3.2-1B-Instruct/en \
  --save-path ./Llama-3.2-1B-Instruct.mxq
```text

**출력:**

- `Llama-3.2-1B-Instruct.mxq` — NPU 실행을 위한 컴파일된 모델

### 대상 디바이스 (`--target-device`)

동일한 스크립트에서 `--target-device`로 대상 NPU를 지정합니다(기본값: `aries-rb`). REGULUS는 `inference_scheme="single"`만 지원하므로 `regulus` 디바이스를 지정하면 자동으로 적용됩니다.

| 사용자 | `--target-device` |
|---|---|
| ARIES | `aries-rb` (기본값) |
| REGULUS (2026-06 이후 고객) | `regulus-rb` |

> **참고:** LLM 컴파일은 신형 REGULUS(`regulus-rb`, 2026-06 이후 고객)에서 지원됩니다. 구형 REGULUS(`regulus-ra`, 2026-06 이전 고객)는 이 워크플로를 지원하지 않습니다.

```bash
# 8-bit (REGULUS)
python generate_mxq.py \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --calib-data-path ./calibration_data/datas/meta-llama-Llama-3.2-1B-Instruct/en \
  --save-path ./Llama-3.2-1B-Instruct.mxq \
  --target-device regulus-rb

# 4-bit (SpinQuant, REGULUS)
python generate_mxq_4bit.py \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --calib-data-path ./calibration_data/datas/meta-llama-Llama-3.2-1B-Instruct/en \
  --save-path ./Llama-3.2-1B-Instruct_w4.mxq \
  --bit w4 \
  --target-device regulus-rb
```text

4-bit 변형은 아래의 임베딩 회전 단계도 필요합니다.

## 다음 단계

컴파일이 끝나면 추론 실행 방법은 [LLM 런타임 튜토리얼](../../runtime/python/llm/README.KR.md)을 참고하세요.

---

## 고급: 4-Bit 양자화

4-bit 양자화는 모델 크기를 더 줄이지만, 정확도 유지를 위해 SpinQuant 회전과 가중치 스케일 탐색이 필요합니다. 이 과정에서 `spinWeight/` 회전 행렬이 생성되며, 추론 전에 임베딩 레이어에 적용해야 합니다.

### 1단계: 4-bit 컴파일

`generate_mxq.py` 대신 `generate_mxq_4bit.py`를 사용합니다:

```bash
python generate_mxq_4bit.py \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --calib-data-path ./calibration_data/datas/meta-llama-Llama-3.2-1B-Instruct/en \
  --save-path ./Llama-3.2-1B-Instruct_w4.mxq \
  --bit w4
```text

- `--bit`: 비트 할당 프리셋. `w4` (전체 4-bit, 기본값) 또는 `w4v8` (value만 8-bit로 유지하여 정확도 향상).

**출력:**

- `Llama-3.2-1B-Instruct_w4.mxq` — 4-bit 컴파일된 모델
- `spinWeight/model/R1/global_rotation.pth` — SpinQuant 회전 행렬

### 2단계: 임베딩 회전

SpinQuant 회전은 모델 내부 가중치 공간을 변환합니다. 임베딩 레이어는 CPU에서 실행되므로 (MXQ로 컴파일되지 않음), 동일한 회전 행렬로 사전 회전해야 합니다.

```bash
python get_rotation_emb.py \
  --embedding-path ./embedding.pt \
  --rotation-matrix-path ./spinWeight/model/R1/global_rotation.pth \
  --output-path ./embedding_rot.pt
```text

**출력:**

- `embedding_rot.pt` — 4-bit 추론용 회전된 임베딩 가중치

> **참고:** 4-bit 모델 추론 시에는 `embedding.pt`가 아닌 `embedding_rot.pt`를 사용하세요. 8-bit 모델은 임베딩 회전이 필요하지 않습니다.
