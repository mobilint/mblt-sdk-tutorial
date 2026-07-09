# STT 런타임

이 튜토리얼은 Mobilint NPU 하드웨어에서 컴파일된 Whisper 음성 인식 모델을 실행하는 방법을 설명합니다.

시작하기 전에 [../../../compilation/stt/README.KR.md](../../../compilation/stt/README.KR.md)의 컴파일 과정을 먼저 완료하세요. 이 디렉토리의 런타임 예제는 다음 파일이 준비되어 있다고 가정합니다.

- `../../../compilation/stt/mxq/whisper-small_encoder.mxq`
- `../../../compilation/stt/mxq/whisper-small_decoder.mxq`
- `../../../compilation/stt/audio_files/`

## 사전 준비

필요한 패키지를 설치하세요.

```bash
pip install -r requirements.txt
```

## 개요

이 튜토리얼은 `mblt-model-zoo`를 사용해 Hugging Face 스타일 API로 Whisper를 실행합니다. 런타임 흐름은 두 단계로 구성됩니다.

1. 컴파일 결과로부터 모델 폴더를 준비합니다.
2. 오디오 파일에 대해 전사 또는 번역을 실행합니다.

준비된 모델 폴더에는 인코더 MXQ, 디코더 MXQ, 임베딩 가중치, generation config, NPU 코어 할당 설정이 함께 저장됩니다.

## 이 튜토리얼의 파일

- `prepare_model.py`: 런타임 추론용 Whisper 모델 폴더를 생성합니다.
- `inference_mblt_model_zoo.py`: `mblt-model-zoo`를 통해 전사 또는 번역을 실행합니다.
- `requirements.txt`: 이 튜토리얼에 필요한 Python 의존성입니다.

## Step 1: 모델 폴더 준비

```bash
python prepare_model.py \
    --encoder-mxq ../../../compilation/stt/mxq/whisper-small_encoder.mxq \
    --decoder-mxq ../../../compilation/stt/mxq/whisper-small_decoder.mxq \
    --output-folder ./whisper-small-mxq \
    --base-model openai/whisper-small
```

이 스크립트는 다음 작업을 수행합니다.

- 컴파일된 인코더와 디코더 MXQ 파일 복사
- 기본 Whisper 설정 다운로드
- 디코더 임베딩 가중치를 `model.safetensors`로 추출
- 기본 NPU 코어 할당이 포함된 `config.json` 작성

## Step 2: 추론 실행

기본 전사 예제를 실행하려면 다음 명령을 사용하세요.

```bash
python inference_mblt_model_zoo.py \
    --audio ../../../compilation/stt/audio_files/en_us_0000.wav \
    --model-folder ./whisper-small-mxq \
    --model-id mobilint/whisper-small
```

자주 쓰는 옵션:

```bash
python inference_mblt_model_zoo.py --audio audio.wav --model-folder ./whisper-small-mxq --model-id mobilint/whisper-small --language en
python inference_mblt_model_zoo.py --audio audio.wav --model-folder ./whisper-small-mxq --model-id mobilint/whisper-small --task translate
```

이 스크립트는 `librosa`로 오디오를 읽고 `16 kHz`로 리샘플링한 뒤, `AutoModelForSpeechSeq2Seq`로 생성을 수행하고 최종 텍스트를 출력합니다.

## NPU 코어 모드

생성된 `config.json`을 편집해 인코더와 디코더의 코어 할당을 바꿀 수 있습니다.

| 모드 | 설명 | 예시 인코더 필드 |
| --- | --- | --- |
| `single` | 단일 코어 실행 | `encoder_target_cores: ["0:0"]` |
| `multi` | 여러 코어가 하나의 추론에 협력 | `encoder_core_mode: "multi"`, `encoder_target_clusters: [0]` |
| `global4` | 한 클러스터를 글로벌 모드로 사용 | `encoder_core_mode: "global4"`, `encoder_target_clusters: [0]` |
| `global8` | 두 클러스터를 글로벌 모드로 사용 | `encoder_core_mode: "global8"`, `encoder_target_clusters: [0, 1]` |

디코더는 같은 패턴을 `decoder_` 접두사로 사용합니다.

## 파라미터

### `prepare_model.py`

- `--encoder-mxq`: 컴파일된 인코더 MXQ 파일 경로
- `--decoder-mxq`: 컴파일된 디코더 MXQ 파일 경로
- `--output-folder`: 준비된 모델 저장 폴더
- `--base-model`: 설정과 임베딩 추출에 사용할 Hugging Face 기본 모델 ID
- `--model-id`: 준비된 설정 파일에 저장할 Hugging Face 모델 ID

### `inference_mblt_model_zoo.py`

- `--audio`: 입력 오디오 파일 경로
- `--model-folder`: 준비된 모델 폴더 경로
- `--model-id`: 프로세서 다운로드에 사용할 Hugging Face 모델 ID
- `--language`: `en`, `ko`, `ja` 같은 선택적 소스 언어 코드
- `--task`: `transcribe` 또는 `translate`

## 예상 출력

스크립트는 generation이 끝난 뒤 최종 전사 또는 번역 결과를 출력합니다.

Whisper는 여러 언어를 지원하며, 예제 스크립트는 `--language`를 통해 언어를 직접 지정하거나 자동 감지를 사용할 수 있습니다.
