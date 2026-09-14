# LLM 런타임

먼저 [LLM 컴파일 튜토리얼](../../../compilation/llm/README.KR.md)을 완료합니다.

## 사전 준비

```bash
pip install -r requirements.txt
```

## `mblt-model-zoo` 추론

`compilation/llm/prepare_model.py`가 생성한 모델 폴더를 사용합니다.

```bash
python inference_mblt_model_zoo.py \
  --model-folder ../../../compilation/llm/llama-mxq-w8
```

주요 옵션:

```bash
python inference_mblt_model_zoo.py --prompt "What is quantum computing?"
python inference_mblt_model_zoo.py --max-new-tokens 512
```

## 직접 `qbruntime` 추론

직접 실행 방식은 MXQ와 준비된 `model.safetensors`를 사용합니다.

```bash
python inference_mxq.py \
  --mxq-path ../../../compilation/llm/Llama-3.2-1B-Instruct-W8.mxq \
  --embedding-path ../../../compilation/llm/llama-mxq-w8/model.safetensors
```

두 스크립트 모두 `--prompt`와 `--max-new-tokens`를 지원합니다.
