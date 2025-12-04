# Whisper 음성-텍스트 변환 런타임

이 튜토리얼은 Mobilint 가속기를 사용하여 컴파일된 Whisper MXQ 모델에서 추론을 실행하는 방법을 설명합니다.

## 사전 요구사항

- [컴파일 튜토리얼](../../../compilation/transformers/stt/README.md)에서 컴파일된 Whisper MXQ 모델 (인코더 및 디코더)
- `maccel` 라이브러리가 포함된 Mobilint SDK
- Python 패키지: `transformers`, `torch`, `librosa`, `safetensors`

## 파일

| 파일 | 설명 |
|------|------|
| `mblt-whisper.py` | HuggingFace 호환 Mobilint Whisper 모델 구현 |
| `prepare_model.py` | 추론을 위한 구성 파일이 포함된 모델 폴더 준비 |
| `inference_mxq.py` | 오디오 전사/번역을 위한 메인 추론 스크립트 |

## 빠른 시작

### 1단계: 모델 폴더 준비

먼저, 컴파일된 MXQ 파일과 필요한 구성이 포함된 모델 폴더를 준비합니다:

```bash
python prepare_model.py \
    --encoder_mxq ../../../compilation/transformers/stt/compilation/compiled/whisper-small_encoder.mxq \
    --decoder_mxq ../../../compilation/transformers/stt/compilation/compiled/whisper-small_decoder.mxq \
    --output_folder ./whisper-small-mxq \
    --base_model openai/whisper-small
```

이 스크립트는:
- 컴파일된 MXQ 파일을 출력 폴더로 복사합니다
- HuggingFace에서 프로세서 (토크나이저 + 특성 추출기)를 다운로드합니다
- 기본 모델에서 임베딩 가중치를 추출하여 저장합니다
- Mobilint Whisper 모델을 위한 적절한 구성을 생성합니다

### 2단계: 추론 실행

오디오 파일에서 음성-텍스트 변환 추론을 실행합니다:

```bash
python inference_mxq.py \
    --audio /path/to/audio.wav \
    --model_folder ./whisper-small-mxq
```

## 사용 옵션

### 기본 전사

```bash
python inference_mxq.py --audio audio.wav --model_folder ./whisper-small-mxq
```

### 언어 지정

```bash
python inference_mxq.py --audio audio.wav --model_folder ./whisper-small-mxq --language en
```

### 영어로 번역

```bash
python inference_mxq.py --audio audio.wav --model_folder ./whisper-small-mxq --task translate
```

### 파이프라인 API 사용

```bash
python inference_mxq.py --audio audio.wav --model_folder ./whisper-small-mxq --use_pipeline
```

## 명령줄 인수

### prepare_model.py

| 인수 | 기본값 | 설명 |
|------|--------|------|
| `--encoder_mxq` | `../../../compilation/transformers/stt/compilation/compiled/whisper-small_encoder.mxq` | 컴파일된 인코더 MXQ 파일 경로 |
| `--decoder_mxq` | `../../../compilation/transformers/stt/compilation/compiled/whisper-small_decoder.mxq` | 컴파일된 디코더 MXQ 파일 경로 |
| `--output_folder` | `./whisper-small-mxq` | 준비된 모델을 위한 출력 폴더 |
| `--base_model` | `openai/whisper-small` | 기본 구성을 위한 HuggingFace 모델 ID |

### inference_mxq.py

| 인수 | 기본값 | 설명 |
|------|--------|------|
| `--audio` | `../../../compilation/transformers/stt/data/audio_files/en_us_0000.wav` | 오디오 파일 경로 |
| `--model_folder` | `./whisper-small-mxq` | 준비된 모델 폴더 경로 |
| `--language` | `None` (자동 감지) | 소스 언어 코드 (예: `en`, `ko`, `ja`) |
| `--task` | `transcribe` | 작업: `transcribe` 또는 `translate` |
| `--use_pipeline` | `False` | 수동 추론 대신 HuggingFace 파이프라인 API 사용 |

## 지원 언어

Whisper는 99개 이상의 언어를 지원합니다. 주요 언어 코드:
- `en` - 영어
- `ko` - 한국어
- `ja` - 일본어
- `zh` - 중국어
- `es` - 스페인어
- `fr` - 프랑스어
- `de` - 독일어

## 아키텍처

이 구현은 원활한 통합을 위해 HuggingFace의 Auto 클래스를 사용합니다:

```
┌─────────────────────────────────────────────────────────┐
│                   inference_mxq.py                      │
│  (AutoModelForSpeechSeq2Seq를 통해 모델 로드)              │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   mblt-whisper.py                       │
│  ┌─────────────────────────────────────────────────┐    │
│  │  MobilintWhisperForConditionalGeneration        │    │
│  │  ├── MobilintWhisperEncoder (encoder.mxq)       │    │
│  │  └── MobilintWhisperDecoder (decoder.mxq)       │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  Mobilint 가속기                         │
│                      (maccel)                           │
└─────────────────────────────────────────────────────────┘
```

## 참고사항

- 오디오 파일은 자동으로 16kHz로 리샘플링됩니다
- 모델은 최대 30초 길이의 오디오 청크를 지원합니다
- 더 긴 오디오의 경우, 파이프라인 API가 자동으로 청킹을 처리합니다
