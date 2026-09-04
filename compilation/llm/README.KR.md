# 대규모 언어 모델(LLM) 컴파일

이 튜토리얼은 [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)를 Mobilint `qbcompiler`로 컴파일하고 런타임 추론용 모델 폴더를 준비합니다.

## 사전 준비

- `qbcompiler`
- Llama 모델 접근 권한이 있는 Hugging Face 계정
- 선택 사항: CUDA GPU

```bash
pip install -r requirements.txt
huggingface-cli login --token <your_huggingface_token>
```

## 1. 캘리브레이션 데이터 생성

캘리브레이션은 Llama 3.2가 공식 지원하는 영어, 독일어, 프랑스어, 이탈리아어, 포르투갈어, 힌디어, 스페인어, 태국어 Wikipedia 텍스트를 모두 사용합니다.
기본 128개 샘플은 8개 언어에 균등하게 배분됩니다.
`--languages`는 캘리브레이션 언어 목록을 변경하며 런타임 언어를 제한하지 않습니다.
한국어 입력도 가능하지만 모델의 공식 지원 언어에는 포함되지 않습니다.

```bash
python generate_calib.py \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --output-dir ./calibration_data
```

## 2. W8 컴파일

```bash
python mxq_compile.py \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --calib-data-path ./calibration_data/datas/meta-llama-Llama-3.2-1B-Instruct/multilingual \
  --save-path ./Llama-3.2-1B-Instruct-W8.mxq \
  --target-device aries-rb
```

### 지원 디바이스

| 디바이스 | 지원 여부 |
| --- | --- |
| `aries-rb` | 지원 |
| `regulus-rb` | 지원 |
| `regulus-ra` | 미지원 |

## 선택 사항: W4V8 컴파일

```bash
python mxq_compile_4bit.py \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --calib-data-path ./calibration_data/datas/meta-llama-Llama-3.2-1B-Instruct/multilingual \
  --save-path ./Llama-3.2-1B-Instruct-W4V8.mxq \
  --target-device aries-rb
```

## 3. 런타임 모델 준비

`prepare_model.py`는 Mobilint Hugging Face 저장소에서 런타임 파일을 받고, `--mxq-path`로 지정한 컴파일된 MXQ로 모델 저장소의 MXQ를 교체한 뒤 `config.json`의 경로를 갱신합니다.

W8은 다음과 같이 실행합니다.

```bash
python prepare_model.py \
  --mxq-path ./Llama-3.2-1B-Instruct-W8.mxq \
  --output-folder ./llama-mxq-w8 \
  --revision W8
```

W4V8은 다음과 같이 실행합니다.

```bash
python prepare_model.py \
  --mxq-path ./Llama-3.2-1B-Instruct-W4V8.mxq \
  --output-folder ./llama-mxq-w4v8 \
  --revision W4V8
```

이후 [LLM 런타임 튜토리얼](../../runtime/python/llm/README.KR.md)을 진행합니다.
