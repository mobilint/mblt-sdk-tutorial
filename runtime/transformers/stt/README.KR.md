# 음성 인식 모델 추론 (Whisper Speech-to-Text)

이 튜토리얼은 Mobilint qbruntime을 사용하여 컴파일된 Whisper 음성 인식 모델로 추론을 실행하는 방법에 대한 단계별 지침을 제공합니다.

이 가이드는 [mblt-sdk-tutorial/compilation/transformers/stt/README.md](file:///workspace/mblt-sdk-tutorial/compilation/transformers/stt/README.md)에서 이어지는 내용입니다. 모델 컴파일을 성공적으로 마쳤다고 가정합니다.

## 사전 요구 사항 (Prerequisites)

추론을 실행하기 전에 다음 구성 요소가 설치되어 있고 준비되었는지 확인하십시오:

- 컴파일된 Whisper MXQ 모델 (인코더 및 디코더)
- `qbruntime` 라이브러리 (NPU 가속기 액세스용)
- Python 패키지: `transformers`, `torch`, `librosa`, `safetensors`

## 파일 목록 (Files)

| 파일 | 설명 |
|------|-------------|
| `mblt-whisper.py` | Hugging Face 호환 Mobilint Whisper 모델 구현체 |
| `prepare_model.py` | 필요한 구성 파일과 함께 모델 디렉토리를 준비하는 스크립트 |
| `inference_mxq.py` | 오디오 전사(transcription) 및 번역을 위한 메인 추론 스크립트 |

## 빠른 시작 (Quick Start)

### 1단계: 모델 폴더 준비

먼저 `prepare_model.py` 스크립트를 실행하여 컴파일된 MXQ 파일을 정리하고 필요한 구성 파일을 생성합니다:

```bash
python prepare_model.py \
    --encoder_mxq ../../../compilation/transformers/stt/compilation/compiled/whisper-small_encoder.mxq \
    --decoder_mxq ../../../compilation/transformers/stt/compilation/compiled/whisper-small_decoder.mxq \
    --output_folder ./whisper-small-mxq \
    --base_model openai/whisper-small
```

이 스크립트는 다음 작업을 수행합니다:
- 컴파일된 MXQ 파일을 지정된 출력 폴더로 복사합니다.
- Hugging Face에서 프로세서(토크나이저 및 특징 추출기)를 다운로드합니다.
- 기본 모델에서 임베딩 가중치를 추출하여 저장합니다.
- Mobilint Whisper 모델에 필요한 구성 파일을 생성합니다.

### 2단계: 추론 실행

`inference_mxq.py`를 사용하여 오디오 파일에 대한 음성 인식 추론을 실행합니다:

```bash
python inference_mxq.py \
    --audio /path/to/audio.wav \
    --model_folder ./whisper-small-mxq
```

## 사용 옵션 (Usage Options)

### 기본 전사 (Basic Transcription)

오디오 파일을 전사하려면:

```bash
python inference_mxq.py --audio audio.wav --model_folder ./whisper-small-mxq
```

### 언어 지정 (Specify Language)

소스 언어를 지정하려면 (예: 영어):

```bash
python inference_mxq.py --audio audio.wav --model_folder ./whisper-small-mxq --language en
```

### 영어로 번역 (Translation to English)

음성 오디오를 영어로 번역하려면:

```bash
python inference_mxq.py --audio audio.wav --model_folder ./whisper-small-mxq --task translate
```

### 파이프라인 API 사용 (Use Pipeline API)

Hugging Face 파이프라인 API를 사용하려면 (긴 오디오 파일에 권장):

```bash
python inference_mxq.py --audio audio.wav --model_folder ./whisper-small-mxq --use_pipeline
```

## 커맨드 라인 인자 (Command Line Arguments)

### `prepare_model.py`

| 인자 | 기본값 | 설명 |
|----------|---------|-------------|
| `--encoder_mxq` | `../../../compilation/transformers/stt/compilation/compiled/whisper-small_encoder.mxq` | 컴파일된 인코더 MXQ 파일 경로 |
| `--decoder_mxq` | `../../../compilation/transformers/stt/compilation/compiled/whisper-small_decoder.mxq` | 컴파일된 디코더 MXQ 파일 경로 |
| `--output_folder` | `./whisper-small-mxq` | 준비된 모델의 저장 폴더 |
| `--base_model` | `openai/whisper-small` | 기본 구성에 사용되는 Hugging Face 모델 ID |

### `inference_mxq.py`

| 인자 | 기본값 | 설명 |
|----------|---------|-------------|
| `--audio` | `../../../compilation/transformers/stt/data/audio_files/en_us_0000.wav` | 입력 오디오 파일 경로 |
| `--model_folder` | `./whisper-small-mxq` | 준비된 모델 폴더 경로 |
| `--language` | `None` (자동 감지) | 소스 언어 코드 (예: `en`, `ko`, `ja`) |
| `--task` | `transcribe` | 수행할 작업: `transcribe` (전사) 또는 `translate` (번역) |
| `--use_pipeline` | `False` | 설정 시, 수동 추론 대신 Hugging Face 파이프라인 API 사용 |

## 지원 언어 (Supported Languages)

Whisper는 99개 이상의 언어를 지원합니다. 주요 언어 코드는 다음과 같습니다:
- `en` - 영어
- `ko` - 한국어
- `ja` - 일본어
- `zh` - 중국어
- `es` - 스페인어
- `fr` - 프랑스어
- `de` - 독일어

## 아키텍처 (Architecture)

이 구현은 원활한 통합을 위해 Hugging Face의 `AutoModel` 클래스를 활용합니다:

```
┌─────────────────────────────────────────────────────────┐
│                   inference_mxq.py                      │
│  (Loads model via AutoModelForSpeechSeq2Seq)            │
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
│                  Mobilint Accelerator                   │
│                      (qbruntime)                        │
└─────────────────────────────────────────────────────────┘
```

## 참고 사항 (Notes)

- 오디오 파일은 자동으로 16kHz로 리샘플링됩니다.
- 모델은 최대 30초 단위의 오디오 청크를 처리합니다.
- 30초보다 긴 오디오 파일의 경우, 청킹(chunking)을 자동으로 처리해주는 파이프라인 API (`--use_pipeline`) 사용을 권장합니다.
