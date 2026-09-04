# 음성-텍스트 변환 Python 런타임

먼저 [STT 컴파일 튜토리얼](../../../compilation/stt/README.KR.md)을 완료합니다. 마지막 `prepare_model.py` 단계가 실행에 필요한 모든 파일을 `compilation/stt/prepared` 아래의 모델 디렉터리 하나에 준비합니다.

모든 명령은 `runtime/python/stt`에서 실행합니다.

## 사전 준비

```bash
pip install -r requirements.txt
```

## 추론

기본 명령은 컴파일 튜토리얼에서 준비한 ARIES 모델과 영어 샘플을 사용합니다.

```bash
python inference_mblt_model_zoo.py
```

REGULUS 모델을 사용하려면 다음과 같이 실행합니다.

```bash
python inference_mblt_model_zoo.py \
  --model-folder ../../../compilation/stt/prepared/regulus-rb/whisper-small
```

다른 오디오, 언어 또는 태스크를 지정할 수 있습니다.

```bash
python inference_mblt_model_zoo.py \
  --audio audio.wav \
  --model-folder ../../../compilation/stt/prepared/aries-rb/whisper-small \
  --language en \
  --task transcribe
```

`--language`을 생략하면 언어를 자동으로 감지합니다. 영어 번역은 `--task translate`를 지정합니다.
