# LLM(대규모 언어 모델) 컴파일

이 튜토리얼은 Mobilint qubee 컴파일러를 사용하여 대규모 언어 모델을 컴파일하는 방법에 대한 상세한 지침을 제공합니다. 컴파일 과정은 표준 트랜스포머 모델을 Mobilint NPU 하드웨어에서 효율적으로 실행할 수 있는 최적화된 `.mxq` 형식으로 변환합니다.

이 튜토리얼에서는 Meta에서 개발한 1B 파라미터 언어 모델인 [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) 모델을 사용합니다.

## 사전 요구 사항

시작하기 전에 다음이 설치 및 준비되어 있는지 확인하십시오.

- qubee SDK 컴파일러 (버전 >= 1.0.1 필요)
- CUDA 지원 GPU (컴파일 시간 단축을 위해 권장)
- Hugging Face 계정 및 Llama 모델 접근 권한 (권한이 필요한 모델 사용 시)

또한, 다음 패키지를 설치해야 합니다.

```bash
pip install accelerate datasets
```

## 개요

컴파일 과정은 세 가지 주요 단계로 구성됩니다.

1.  **모델 준비**: 모델 다운로드 및 임베딩 가중치 추출
2.  **캘리브레이션 데이터셋 생성**: Wikitext 데이터셋에서 캘리브레이션 데이터 생성
3.  **모델 컴파일**: 캘리브레이션 데이터를 사용하여 모델을 `.mxq` 형식으로 변환

## Step 1: 모델 준비

모델을 사용하기 전에 [Hugging Face](https://huggingface.co/)에 계정을 등록하고 [모델 페이지](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)에서 라이선스 동의서에 동의해야 합니다.

그 후 다음 명령을 사용하여 Hugging Face에 로그인하고, `<your_huggingface_token>`을 실제 토큰으로 교체하십시오.

```bash
hf auth login --token <your_huggingface_token>
```

토큰은 [Hugging Face 계정 설정](https://huggingface.co/settings/tokens)에서 확인할 수 있습니다.

다음으로 모델을 다운로드하고 임베딩 레이어 가중치를 추출합니다. 임베딩 레이어는 런타임 중에 별도로 처리되며, 나머지 모델은 NPU에서 실행됩니다.

```bash
python download_model.py \
  --repo_id meta-llama/Llama-3.2-1B-Instruct \
  --embedding ./embedding.pt
```

**수행 내용:**

- Hugging Face Hub에서 지정된 모델을 다운로드하여 캐시 디렉토리에 저장합니다.
- 입력 임베딩 레이어 가중치를 추출합니다.
- 임베딩 가중치를 `embedding.pt`에 저장합니다.

**매개변수:**

- `--repo_id`: Hugging Face 모델 식별자
- `--embedding`: 임베딩 가중치 파일의 출력 경로

## Step 2: 캘리브레이션 데이터셋 준비

캘리브레이션 데이터는 컴파일 중 양자화에 필수적입니다. [Wikipedia 기사](https://huggingface.co/datasets/wikimedia/wikipedia)에서 텍스트를 일반적인 모델 입력을 나타내는 임베딩 벡터로 변환하여 데이터를 생성합니다.

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

**수행 내용:**

- 지정된 언어에 대한 Wikitext 데이터셋을 로드합니다.
- 모델 토크나이저를 사용하여 텍스트 샘플을 토크나이징합니다.
- 추출된 임베딩 레이어를 사용하여 토큰을 임베딩으로 변환합니다.
- 캘리브레이션 샘플을 `.npy` 파일로 저장합니다.

**매개변수:**

- `--model_tag`: 모델 식별자 (출력 디렉토리 이름 지정에 사용)
- `--embedding_path`: Step 1에서 추출한 임베딩 가중치 경로
- `--tokenizer_path`: Hugging Face 토크나이저 식별자
- `--output_dir`: 캘리브레이션 데이터의 기본 디렉토리
- `--min_seqlen`: 최소 시퀀스 길이 (이보다 짧은 샘플은 제외)
- `--max_seqlen`: 최대 시퀀스 길이 (샘플이 이 길이로 잘림)
- `--max_calib`: 언어당 생성할 캘리브레이션 샘플 수

**출력 위치:**
캘리브레이션 파일은 다음 위치에 저장됩니다. `./calib/datas/meta-llama-Llama-3.2-1B-Instruct/en/`

**다국어 지원:**
여러 언어를 사용하려면 `generate_calib.py`에서 `LANGUAGES` 목록을 수정하십시오.

```python
LANGUAGES = ["en", "de", "fr", "es", "it", "ja", "ko", "zh"] # 추가 언어 지원 가능
```

여러 언어를 사용하는 경우, 컴파일 전에 각 언어 디렉토리를 하나의 디렉토리로 병합하십시오.

## Step 3: 모델 컴파일

모델과 캘리브레이션 데이터셋이 준비되면 모델을 `.mxq` 형식으로 컴파일합니다. 이 과정에서 NPU 실행을 위한 양자화 및 최적화가 수행됩니다.

```bash
python generate_mxq.py \
  --model_path meta-llama/Llama-3.2-1B-Instruct \
  --calib_data_path ./calib/datas/meta-llama-Llama-3.2-1B-Instruct/en \
  --save_path ./Llama-3.2-1B-Instruct.mxq
```

**수행 내용:**

- Hugging Face에서 원본 모델을 로드합니다.
- 캘리브레이션 데이터를 사용하여 최적의 양자화 매개변수를 결정합니다.
- NPU 실행을 위해 모델 레이어를 컴파일합니다.
- 컴파일된 모델을 `.mxq` 형식으로 저장합니다.

**매개변수:**

- `--model_path`: Hugging Face 모델 식별자
- `--calib_data_path`: Step 2에서 생성한 캘리브레이션 데이터 디렉토리 경로
- `--save_path`: 컴파일된 `.mxq` 모델의 출력 경로

**예상 출력:**
컴파일된 모델은 `./Llama-3.2-1B-Instruct.mxq`로 저장됩니다.

## 다음 단계

컴파일이 성공적으로 완료되면, 컴파일된 모델로 추론을 실행하는 방법에 대한 지침이 담긴 `mblt-sdk-tutorial/runtime/transformers/llm/README.KR.md`를 참조하십시오.

## 파일 요약

- `embedding.pt` - 추출된 임베딩 레이어 가중치
- `Llama-3.2-1B-Instruct.mxq` - NPU 실행을 위한 컴파일된 모델
- `calib/datas/meta-llama-Llama-3.2-1B-Instruct/en/` - 캘리브레이션 데이터셋 (128개 샘플)

## 추가 정보: 고급 양자화를 사용한 모델 컴파일

qubee SDK 1.0.0 출시 이후, 모델에 대한 고급 양자화 기술을 지원합니다. 이를 통해 유연한 비트 할당과 고급 양자화 매개변수 최적화를 사용하여 더 나은 성능을 얻을 수 있습니다.

`generate_mxq.py` 스크립트에는 W4V8, W4 등 모델 레이어에 대한 비트 할당이 사전 설정되어 있습니다.

다음과 같이 추가 파라미터를 사용하여 이 기능을 활성화할 수 있습니다.

```bash
python generate_mxq.py \
  --model_path meta-llama/Llama-3.2-1B-Instruct \
  --calib_data_path ./calib/datas/meta-llama-Llama-3.2-1B-Instruct/en \
  --save_path ./Llama-3.2-1B-Instruct.mxq \
  --bit w4
```

**매개변수:**

- `--bit`: 모델 레이어의 비트 할당. `w4`, `w4v8`, `w8` 옵션이 가능합니다.

### Spin Quant 및 Rotation Transform

더 나은 정확도를 위해 회전 변환(Rotation Transform)이 포함된 Spin Quant 사용을 적극 권장합니다. 제공된 예제 코드는 `w4` 또는 `w4v8` 양자화 선택 시 이를 자동으로 사용하도록 설계되었습니다.

`w4` 또는 `w4v8` 양자화로 `generate_mxq.py`를 실행하면 회전 행렬이 생성됩니다(예: `spinWeight/model/R1/global_rotation.pth`). 추론 시에는 이 회전 행렬을 사용하여 임베딩 행렬을 회전시켜야 합니다.

다음 명령어를 실행하여 임베딩 행렬을 회전시키십시오.

```bash
python rotate_embedding.py \
  --embedding_path ./embedding.pt \
  --rotation_matrix_path ./spinWeight/model/R1/global_rotation.pth \
  --output_path ./embedding_rotated.pt
```

**참고:** `w8` 양자화를 사용하는 경우 임베딩 행렬을 회전시킬 필요가 없습니다. `w4` 및 `w4v8` 양자화의 경우, 추론 시 반드시 회전된 임베딩 행렬을 사용해야 합니다.
