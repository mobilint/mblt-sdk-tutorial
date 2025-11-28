# 대규모 언어 모델 컴파일

이 튜토리얼은 Mobilint qubee 컴파일러를 사용하여 대규모 언어 모델을 컴파일하는 방법에 대한 상세한 지침을 제공합니다. 컴파일 과정은 표준 트랜스포머 모델을 Mobilint NPU 하드웨어에서 효율적으로 실행할 수 있는 최적화된 `.mxq` 형식으로 변환합니다.

이 튜토리얼에서는 Meta에서 개발한 10억 파라미터 언어 모델인 [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) 모델을 사용합니다.

## 사전 요구사항

시작하기 전에 다음이 설치되어 있는지 확인하세요:

- qubee SDK 컴파일러 설치 (버전 >= 0.11 필요)
- CUDA 지원 GPU (컴파일 시간 단축을 위해 권장)
- Llama 모델에 대한 접근 권한이 있는 HuggingFace 계정 (접근 권한이 필요한 모델 사용 시)

또한, 다음 패키지를 설치해야 합니다:

```bash
pip install accelerate datasets
```

## 개요

컴파일 과정은 세 가지 주요 단계로 구성됩니다:

1. **모델 준비**: 모델 다운로드 및 임베딩 가중치 추출
2. **캘리브레이션 데이터셋 생성**: Wikitext 데이터셋에서 캘리브레이션 데이터 생성
3. **모델 컴파일**: 캘리브레이션 데이터를 사용하여 모델을 `.mxq` 형식으로 변환

## Step 1: 모델 준비

모델을 사용하기 전에 [HuggingFace](https://huggingface.co/)에 계정을 등록하고 [모델 페이지](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)에서 모델 사용 동의서에 동의해야 합니다.

그런 다음 다음 명령을 사용하여 HuggingFace에 로그인하고 `<your_huggingface_token>`을 실제 HuggingFace 토큰으로 교체해야 합니다:

```bash
hf auth login --token <your_huggingface_token>
```

HuggingFace 토큰에 대해 알 수 없는 경우 [HuggingFace 계정 설정](https://huggingface.co/settings/tokens)에서 찾을 수 있습니다.

그런 다음 HuggingFace에서 모델을 다운로드하고 임베딩 레이어 가중치를 추출해야 합니다. 임베딩 레이어는 런타임 중에 별도로 사용되며, 나머지 모델은 NPU에서 실행됩니다.

```bash
python download_model.py \
  --repo_id meta-llama/Llama-3.2-1B-Instruct \
  --embedding ./embedding.pt
```

**이 작업의 내용:**

- HuggingFace Hub에서 지정된 모델을 다운로드하고 캐시 디렉토리에 저장
- 입력 임베딩 레이어 가중치 추출
- 임베딩 가중치를 `embedding.pt`에 저장

**매개변수:**

- `--repo_id`: HuggingFace 모델 식별자
- `--embedding`: 임베딩 가중치 파일의 출력 경로

## Step 2: 캘리브레이션 데이터셋 준비

캘리브레이션 데이터는 컴파일 중 양자화에 필수적입니다. [Wikipedia 기사](https://huggingface.co/datasets/wikimedia/wikipedia)에서 이 데이터를 생성하여 텍스트를 일반적인 모델 입력을 나타내는 임베딩 벡터로 변환합니다.

```bash
python generate_calib.py \
  --model_tag meta-llama/Llama-3.2-1B-Instruct \
  --embedding_path ./embedding.pt \
  --tokenizer_path meta-llama/Llama-3.2-1B-Instruct \
  --output_dir ./calib \
  --min_seqlen 512 \
  --max_seqlen 2048 \
  --max_calib 128
```

**이 작업의 내용:**

- 지정된 언어에 대한 Wikitext 데이터셋 로드
- 모델의 토크나이저를 사용하여 텍스트 샘플 토크나이징
- 추출된 임베딩 레이어를 사용하여 토큰을 임베딩으로 변환
- 캘리브레이션 샘플을 `.npy` 파일로 저장

**매개변수:**

- `--model_tag`: 모델 식별자 (출력 디렉토리 명명에 사용)
- `--embedding_path`: Step 1에서 추출한 임베딩 가중치 경로
- `--tokenizer_path`: HuggingFace 토크나이저 식별자
- `--output_dir`: 캘리브레이션 데이터의 기본 디렉토리
- `--min_seqlen`: 최소 시퀀스 길이 (이보다 짧은 샘플은 건너뜀)
- `--max_seqlen`: 최대 시퀀스 길이 (샘플이 이 길이로 잘림)
- `--max_calib`: 언어당 생성할 캘리브레이션 샘플 수

**출력 위치:**
캘리브레이션 파일은 다음 위치에 저장됩니다: `./calib/datas/meta-llama-Llama-3.2-1B-Instruct/en/`

**다국어 지원:**
여러 언어를 사용하려면 `generate_calib.py`의 `LANGUAGES` 목록을 수정하세요:

```python
LANGUAGES = ["en", "de", "fr", "es", "it", "ja", "ko", "zh"] # 더 많은 언어가 지원됩니다
```

여러 언어를 사용하는 경우 컴파일 전에 언어 디렉토리를 단일 디렉토리로 병합하세요.

## Step 3: 모델 컴파일

모델과 캘리브레이션 데이터셋을 준비한 후 모델을 `.mxq` 형식으로 컴파일하세요. 이 과정은 NPU 실행을 위한 양자화 및 최적화를 수행합니다.

```bash
python generate_mxq.py \
  --model_path meta-llama/Llama-3.2-1B-Instruct \
  --calib_data_path ./calib/datas/meta-llama-Llama-3.2-1B-Instruct/en \
  --save_path ./Llama-3.2-1B-Instruct.mxq
```

**이 작업의 내용:**

- HuggingFace에서 원본 모델 로드
- 캘리브레이션 데이터를 사용하여 최적의 양자화 매개변수 결정
- NPU 실행을 위해 모델 레이어 컴파일
- 컴파일된 모델을 `.mxq` 형식으로 저장

**매개변수:**

- `--model_path`: HuggingFace 모델 식별자
- `--calib_data_path`: Step 2에서 생성한 캘리브레이션 데이터 디렉토리 경로
- `--save_path`: 컴파일된 `.mxq` 모델의 출력 경로

**예상 출력:**
컴파일된 모델은 `./Llama-3.2-1B-Instruct.mxq`로 저장됩니다.

## 다음 단계

컴파일이 성공적으로 완료된 후, 컴파일된 모델로 추론을 실행하는 방법에 대한 지침은 `mblt-sdk-tutorial/runtime/transformers/llm/README.md`를 참조하세요.

## 파일 요약

- `embedding.pt` - 추출된 임베딩 레이어 가중치
- `Llama-3.2-1B-Instruct.mxq` - NPU 실행을 위한 컴파일된 모델
- `calib/datas/meta-llama-Llama-3.2-1B-Instruct/en/` - 캘리브레이션 데이터셋 (128개 샘플)
