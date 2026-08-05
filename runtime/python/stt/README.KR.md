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

1. 자체 포함형 모델 폴더를 준비합니다(HF repo를 다운로드한 뒤 컴파일한 MXQ로 교체).
2. 오디오 파일에 대해 전사 또는 번역을 실행합니다.

준비된 폴더는 자체 포함형입니다. `config.json`, 번들 proxy 클래스, tokenizer/processor, generation config, 인코더/디코더 MXQ 파일, 그리고 `model.safetensors`(디코더 임베딩 가중치)를 함께 담습니다.

## 이 튜토리얼의 파일

- `prepare_model.py`: 런타임 추론용 Whisper 모델 폴더를 생성합니다.
- `inference_mblt_model_zoo.py`: `mblt-model-zoo`를 통해 전사 또는 번역을 실행합니다.
- `requirements.txt`: 이 튜토리얼에 필요한 Python 의존성입니다.

## Step 1: 모델 폴더 준비

```bash
python prepare_model.py \
    --repo-id mobilint/whisper-small \
    --compilation-dir ../../../compilation/stt/mxq \
    --output-folder ./whisper-small-mxq \
    --force
```

이 스크립트는 다음 작업을 수행합니다.

- `huggingface_hub.snapshot_download`로 Hugging Face repo 다운로드(자체 포함형 `config.json`, proxy 클래스, tokenizer, generation config, `model.safetensors`), repo 자체의 `.mxq`만 제외
- 컴파일한 인코더와 디코더 `.mxq`를 컴파일 디렉터리에서 복사
- `config.json`의 `encoder_mxq_path` / `decoder_mxq_path`를 복사한 파일명으로 패치(repo의 코어 할당은 유지)

> `git-lfs`가 필요 없습니다 — `snapshot_download`가 실제 파일을 받습니다(`huggingface_hub`는 `mblt-model-zoo[transformers]`와 함께 설치됨).

## Step 2: 추론 실행

기본 전사 예제를 실행하려면 다음 명령을 사용하세요.

```bash
python inference_mblt_model_zoo.py \
    --audio ../../../compilation/stt/audio_files/en_us_0000.wav \
    --model-folder ./whisper-small-mxq
```

자주 쓰는 옵션:

```bash
python inference_mblt_model_zoo.py --audio audio.wav --model-folder ./whisper-small-mxq --language en
python inference_mblt_model_zoo.py --audio audio.wav --model-folder ./whisper-small-mxq --task translate
```

이 스크립트는 `soundfile`로 `16 kHz` 모노 오디오를 읽고, `AutoModelForSpeechSeq2Seq`로 생성을 수행하고 최종 텍스트를 출력합니다.

## NPU 코어 모드

모델 폴더의 `config.json`(다운로드한 repo 기준, 기본값 `global8`)을 편집해 인코더와 디코더의 코어 할당을 바꿀 수 있습니다.

| 모드 | 설명 | 예시 인코더 필드 |
| --- | --- | --- |
| `single` | 단일 코어 실행 | `encoder_target_cores: ["0:0"]` |
| `multi` | 여러 코어가 하나의 추론에 협력 | `encoder_core_mode: "multi"`, `encoder_target_clusters: [0]` |
| `global4` | 한 클러스터를 글로벌 모드로 사용 | `encoder_core_mode: "global4"`, `encoder_target_clusters: [0]` |
| `global8` | 두 클러스터를 글로벌 모드로 사용 | `encoder_core_mode: "global8"`, `encoder_target_clusters: [0, 1]` |

디코더는 같은 패턴을 `decoder_` 접두사로 사용합니다.

## 파라미터

### `prepare_model.py`

- `--repo-id`: 다운로드할 Hugging Face repo id(자체 포함형 config, proxy 클래스, tokenizer, 임베딩)
- `--compilation-dir`: 2개의 `.mxq`(인코더와 디코더)가 있는 컴파일 출력 디렉터리 경로
- `--output-folder`: 대상 폴더(다운로드한 repo에 컴파일한 MXQ를 교체해 넣음)
- `--force`: 대상 폴더가 이미 있으면 먼저 삭제

### `inference_mblt_model_zoo.py`

- `--audio`: 입력 오디오 파일 경로
- `--model-folder`: 준비된 모델 폴더 경로
- `--language`: `en`, `ko`, `ja` 같은 선택적 소스 언어 코드
- `--task`: `transcribe` 또는 `translate`

## 예상 출력

스크립트는 generation이 끝난 뒤 최종 전사 또는 번역 결과를 출력합니다.

Whisper는 여러 언어를 지원하며, 예제 스크립트는 `--language`를 통해 언어를 직접 지정하거나 자동 감지를 사용할 수 있습니다.
