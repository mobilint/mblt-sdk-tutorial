# Speech-to-Text Python Runtime

Complete the [STT compilation tutorial](../../../compilation/stt/README.md) first. Its final `prepare_model.py` step creates one self-contained model directory under `compilation/stt/prepared`.

Run all commands from `runtime/python/stt`.

## Prerequisites

```bash
pip install -r requirements.txt
```

## Inference

The default command uses the ARIES model and English sample prepared by the compilation tutorial.

```bash
python inference_mblt_model_zoo.py
```

To use a REGULUS model:

```bash
python inference_mblt_model_zoo.py \
  --model-folder ../../../compilation/stt/prepared/regulus-rb/whisper-small
```

To select another audio file, language, or task:

```bash
python inference_mblt_model_zoo.py \
  --audio audio.wav \
  --model-folder ../../../compilation/stt/prepared/aries-rb/whisper-small \
  --language en \
  --task transcribe
```

Omit `--language` to use automatic language detection. Set `--task translate` to translate speech into English.
