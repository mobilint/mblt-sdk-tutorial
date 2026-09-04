# Vision-Language 모델 런타임

준비된 `Qwen3-VL-2B-Instruct` 모델을 `mblt-model-zoo`로 실행합니다.

## 사전 준비

```bash
pip install -r requirements.txt
```

먼저 [VLM 컴파일 튜토리얼](../../../compilation/vlm/README.KR.md)의 `prepare_model.py` 단계까지 완료합니다.

## 추론 실행

기본 명령은 `compilation/vlm/prepared/aries-rb/Qwen3-VL-2B-Instruct`에 준비된 ARIES 모델을 사용합니다.

```bash
python inference_mblt_model_zoo.py
```

REGULUS 모델을 사용하려면 모델 폴더를 지정합니다.

```bash
python inference_mblt_model_zoo.py \
  --model-folder ../../../compilation/vlm/prepared/regulus-rb/Qwen3-VL-2B-Instruct
```

로컬 이미지나 URL과 프롬프트를 직접 지정할 수도 있습니다.

```bash
python inference_mblt_model_zoo.py \
  --image /path/to/image.jpg \
  --prompt "이 이미지에 어떤 물체가 있나요?"
```
