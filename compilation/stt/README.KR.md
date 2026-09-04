# 음성-텍스트 변환 모델 컴파일

이 튜토리얼은 [OpenAI Whisper Small](https://huggingface.co/openai/whisper-small)의 인코더와 디코더를 MXQ로 컴파일하고, 실행에 필요한 파일을 하나의 런타임 모델 디렉터리에 준비합니다.

모든 명령은 `compilation/stt`에서 실행합니다.

## 사전 준비

```bash
pip install -r requirements.txt
```

Whisper parser에는 `transformers==4.50.0`이 필요합니다. `requirements.txt`에 해당 버전이 고정되어 있습니다.

## 지원 디바이스

| 디바이스 | 지원 여부 |
| --- | --- |
| `aries-rb` | 지원 |
| `regulus-rb` | 지원 |
| `regulus-ra` | 미지원 |

## 1. 오디오 데이터 준비

```bash
python prepare_audio.py
```

FLEURS 검증 데이터에서 17개 언어별로 20개씩 내려받아 16 kHz PCM WAV와 전사문 340쌍을 `./audio_files`에 저장합니다.

```text
audio_files/
├── English/
│   ├── en_us_0000.wav
│   └── en_us_0000.txt
├── Korean/
└── ...
```

## 2. 캘리브레이션 데이터 생성

```bash
python generate_calibration.py
```

인코더에는 오디오별 log-mel spectrogram을 사용합니다. 디코더에는 다섯 가지 오디오 길이를 사용하고, Whisper 언어·태스크 prefix를 복원한 뒤 마지막 EOS와 8토큰 미만 샘플을 제외합니다.

결과는 `./calibration_data`에 저장됩니다.

```text
calibration_data/
├── encoder/
│   ├── whisper_encoder_cali.txt
│   └── encoder_calib_*.npy
└── decoder/
    ├── whisper_decoder_calib.json
    └── sample_*/
        ├── decoder_hidden_states.npy
        └── encoder_hidden_states.npy
```

생성하기 전에 출력 디렉터리가 비어 있어야 합니다.

## 3. MXQ 컴파일

ARIES:

```bash
python compile_encoder.py --target-device aries-rb
python compile_decoder.py --target-device aries-rb
```

REGULUS:

```bash
python compile_encoder.py --target-device regulus-rb
python compile_decoder.py --target-device regulus-rb
```

각 스크립트는 대상 디바이스별 MBLT를 다시 만든 후 MXQ를 컴파일합니다.

```text
mblt/<target-device>/whisper-small_{encoder,decoder}.mblt
mxq/<target-device>/whisper-small_{encoder,decoder}.mxq
```

검증된 Whisper 실험의 컴파일 설정은 자동으로 적용됩니다.

- ARIES는 `inference_scheme="all"`, REGULUS는 `inference_scheme="single"`을 사용합니다.
- 인코더와 디코더에 검증된 equivalent transformation과 mixed-precision activation 설정을 사용합니다.
- 디코더에는 max calibration, full-sequence LLM calibration, Hessian quantization을 추가로 사용합니다.
- REGULUS 디코더의 sequence와 cache 길이는 1024로 제한합니다.

## 4. 런타임 모델 준비

인코더와 디코더 MXQ 컴파일이 끝난 뒤 실행합니다.

```bash
python prepare_model.py --target-device aries-rb
```

REGULUS:

```bash
python prepare_model.py --target-device regulus-rb
```

결과는 `./prepared/<target-device>/whisper-small`에 저장됩니다. 이 디렉터리 하나에 런타임에 필요한 processor, tokenizer, 설정, CPU embedding weight와 두 MXQ 파일이 모두 들어갑니다.

출력 디렉터리가 이미 존재하면 `--force`를 지정해 교체합니다.

## 런타임

[Python STT 런타임 튜토리얼](../../runtime/python/stt/README.KR.md)을 계속 진행합니다.
