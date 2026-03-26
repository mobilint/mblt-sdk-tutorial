# LLM(대규모 언어 모델) 컴파일

이 튜토리얼은 Mobilint qbcompiler를 사용하여 대규모 언어 모델을 컴파일하는 방법을 설명합니다.

이 튜토리얼에서는 Meta에서 개발한 1B 파라미터 언어 모델인 [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) 모델을 사용합니다.

## 개요

컴파일 과정은 세 가지 주요 단계로 구성됩니다:

1. **모델 준비**: 모델 다운로드 및 임베딩 가중치 추출
2. **캘리브레이션 데이터 생성**: Wikipedia 기사에서 캘리브레이션 데이터 생성
3. **모델 컴파일**: 8-bit 양자화로 모델을 `.mxq` 형식으로 변환

## 사전 준비

- qbcompiler SDK (버전 >= 1.0.1)
- (선택) CUDA GPU (컴파일 시간 단축)
- Hugging Face 계정 및 Llama 모델 접근 권한

```bash
pip install -r requirements.txt
```

## Step 1: 모델 다운로드

[Hugging Face](https://huggingface.co/)에 가입하고 [모델 페이지](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)에서 라이선스에 동의한 후 로그인합니다:

```bash
huggingface-cli login --token <your_huggingface_token>
```

모델을 다운로드하고 임베딩 가중치를 추출합니다. 임베딩 레이어는 추론 시 CPU에서 실행되며, 나머지 모델은 NPU에서 실행됩니다.

```bash
python download_model.py \
  --repo-id meta-llama/Llama-3.2-1B-Instruct \
  --embedding-path ./embedding.pt
```

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
```

**출력:**

- `./calibration_data/datas/meta-llama-Llama-3.2-1B-Instruct/en/` — 128개 캘리브레이션 샘플 (`.npy`)

## Step 3: 모델 컴파일 (8-bit)

8-bit 양자화로 모델을 `.mxq` 형식으로 컴파일합니다.

```bash
python generate_mxq.py \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --calib-data-path ./calibration_data/datas/meta-llama-Llama-3.2-1B-Instruct/en \
  --save-path ./Llama-3.2-1B-Instruct.mxq
```

**출력:**

- `Llama-3.2-1B-Instruct.mxq` — NPU 실행을 위한 컴파일된 모델

## 다음 단계

컴파일 완료 후, 추론 실행 방법은 `mblt-sdk-tutorial/runtime/transformers/llm/README.KR.md`를 참조하세요.

---

## Advanced: 4-Bit 양자화

4-bit 양자화는 모델 크기를 더 줄이지만, 정확도 유지를 위해 SpinQuant 회전과 가중치 스케일 탐색이 필요합니다. 이 과정에서 `spinWeight/` 회전 행렬이 생성되며, 추론 전에 임베딩 레이어에 적용해야 합니다.

### Step 1: 4-bit 컴파일

`generate_mxq.py` 대신 `generate_mxq_4bit.py`를 사용합니다:

```bash
python generate_mxq_4bit.py \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --calib-data-path ./calibration_data/datas/meta-llama-Llama-3.2-1B-Instruct/en \
  --save-path ./Llama-3.2-1B-Instruct_w4.mxq \
  --bit w4
```

- `--bit`: 비트 할당 프리셋. `w4` (전체 4-bit, 기본값) 또는 `w4v8` (value만 8-bit로 유지하여 정확도 향상).

**출력:**

- `Llama-3.2-1B-Instruct_w4.mxq` — 4-bit 컴파일된 모델
- `spinWeight/model/R1/global_rotation.pth` — SpinQuant 회전 행렬

### Step 2: 임베딩 회전

SpinQuant 회전은 모델 내부 가중치 공간을 변환합니다. 임베딩 레이어는 CPU에서 실행되므로 (MXQ로 컴파일되지 않음), 동일한 회전 행렬로 사전 회전해야 합니다.

```bash
python get_rotation_emb.py \
  --embedding-path ./embedding.pt \
  --rotation-matrix-path ./spinWeight/model/R1/global_rotation.pth \
  --output-path ./embedding_rot.pt
```

**출력:**

- `embedding_rot.pt` — 4-bit 추론용 회전된 임베딩 가중치

> **참고:** 4-bit 모델 추론 시에는 `embedding.pt`가 아닌 `embedding_rot.pt`를 사용하세요. 8-bit 모델은 임베딩 회전이 필요하지 않습니다.
