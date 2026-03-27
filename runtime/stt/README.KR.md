# 음성 인식 (STT) 모델 추론

이 튜토리얼은 Mobilint NPU 하드웨어에서 컴파일된 Whisper 음성 인식 모델로 추론을 실행하는 방법을 안내합니다.

이 튜토리얼에서는 [컴파일 튜토리얼](../../compilation/stt/README.md)에서 컴파일한 [Whisper Small](https://huggingface.co/mobilint/whisper-small) 모델을 사용합니다. **먼저 컴파일 튜토리얼을 완료해야** 다음 파일들이 준비됩니다:

- `compilation/stt/mxq/whisper-small_encoder.mxq` - 컴파일된 인코더
- `compilation/stt/mxq/whisper-small_decoder.mxq` - 컴파일된 디코더
- `compilation/stt/audio_files/` - 테스트용 오디오 샘플

## 개요

추론 과정은 두 단계로 구성됩니다:

1. **모델 준비**: MXQ 파일 정리 및 필요한 구성 파일 다운로드
2. **추론 실행**: mblt-model-zoo를 사용하여 오디오 파일에 대한 음성 인식 수행

이 튜토리얼은 [mblt-model-zoo](https://docs.mobilint.com/model-zoo)를 사용하여 간편한 추론 방식을 제공합니다. mblt-model-zoo를 통해 컴파일된 MXQ 모델을 한 줄(`AutoModelForSpeechSeq2Seq.from_pretrained()`)로 로드할 수 있으며, 일반 HuggingFace 모델과 동일한 방식으로 사용할 수 있습니다. NPU 코어 할당, KV cache, 리소스 관리는 자동으로 처리됩니다.

모든 스크립트는 `runtime/stt/` 디렉토리에서 실행합니다.

## 사전 요구 사항

```bash
pip install -r requirements.txt
```

## Step 1: 모델 폴더 준비

컴파일된 MXQ 파일을 정리하고 HuggingFace에서 필요한 구성 파일을 다운로드합니다.

```bash
python prepare_model.py \
    --encoder-mxq ../../compilation/stt/mxq/whisper-small_encoder.mxq \
    --decoder-mxq ../../compilation/stt/mxq/whisper-small_decoder.mxq \
    --output-folder ./whisper-small-mxq \
    --base-model openai/whisper-small
```

**수행 내용:**

- 컴파일된 MXQ 파일을 출력 폴더로 복사
- 기본 모델에서 임베딩 가중치 추출 및 저장
- mblt-model-zoo용 NPU 코어 할당 설정이 포함된 `config.json` 생성

**출력:**

- `./whisper-small-mxq/` - 추론 준비가 완료된 모델 폴더

## Step 2: mblt-model-zoo를 사용한 추론

오디오 파일에 대한 음성 인식 추론을 실행합니다. 이 스크립트는 mblt-model-zoo의 공식 Whisper 구현을 사용하여 HuggingFace `AutoModel` API로 MXQ 모델을 로드합니다.

```bash
python inference_mblt_model_zoo.py \
    --audio ../../compilation/stt/audio_files/en_us_0000.wav \
    --model-folder ./whisper-small-mxq \
    --model-id mobilint/whisper-small
```

**추가 옵션:**

```bash
# 소스 언어 지정 (예: 영어)
python inference_mblt_model_zoo.py --audio audio.wav --model-folder ./whisper-small-mxq --model-id mobilint/whisper-small --language en

# 음성을 영어로 번역
python inference_mblt_model_zoo.py --audio audio.wav --model-folder ./whisper-small-mxq --model-id mobilint/whisper-small --task translate
```

## NPU 추론 모드

NPU는 다양한 코어 모드를 지원합니다. 코어 모드는 `config.json`(`prepare_model.py`가 생성)에서 설정합니다.

| 모드 | 설명 | config.json 필드 |
|------|------|-----------------|
| `single` | 각 코어가 독립적으로 추론 실행. 기본 모드. | `encoder_target_cores: ["0:0"]` |
| `multi` | 여러 코어가 하나의 추론에 협력. | `encoder_core_mode: "multi"`, `encoder_target_clusters: [0]` |
| `global4` | 4개 코어(1 클러스터)가 글로벌 모드로 실행. | `encoder_core_mode: "global4"`, `encoder_target_clusters: [0]` |
| `global8` | 8개 코어(2 클러스터)가 글로벌 모드로 실행. 최대 처리량. | `encoder_core_mode: "global8"`, `encoder_target_clusters: [0, 1]` |

디코더도 동일한 필드를 `decoder_` 접두사로 사용합니다 (예: `decoder_target_cores`, `decoder_core_mode`).

**예시: global8 모드로 변경**

`whisper-small-mxq/config.json`을 편집:
```json
{
    "encoder_core_mode": "global8",
    "encoder_target_clusters": [0, 1],
    "decoder_core_mode": "global8",
    "decoder_target_clusters": [0, 1]
}
```

> **참고:** `target_cores` 형식은 `"cluster:core"`입니다 (예: `"0:0"` = 클러스터 0, 코어 0). multi/global4/global8 사용 시에는 `target_cores` 대신 `target_clusters`를 사용합니다.

## 커맨드 라인 인자

### `prepare_model.py`

| 인자 | 기본값 | 설명 |
| --- | ----- | --- |
| `--encoder-mxq` | `../../compilation/stt/mxq/whisper-small_encoder.mxq` | 컴파일된 인코더 MXQ 파일 경로 |
| `--decoder-mxq` | `../../compilation/stt/mxq/whisper-small_decoder.mxq` | 컴파일된 디코더 MXQ 파일 경로 |
| `--output-folder` | `./whisper-small-mxq` | 준비된 모델의 저장 폴더 |
| `--base-model` | `openai/whisper-small` | 기본 구성 및 임베딩 추출용 HuggingFace 모델 ID |
| `--model-id` | `mobilint/whisper-small` | mblt-model-zoo용 HuggingFace 모델 ID |

### `inference_mblt_model_zoo.py`

| 인자 | 기본값 | 설명 |
| --- | ----- | --- |
| `--audio` | `../../compilation/stt/audio_files/en_us_0000.wav` | 입력 오디오 파일 경로 |
| `--model-folder` | `./whisper-small-mxq` | 준비된 모델 폴더 경로 |
| `--model-id` | `mobilint/whisper-small` | 프로세서 다운로드용 HuggingFace 모델 ID |
| `--language` | `None` (자동 감지) | 소스 언어 코드 (예: `en`, `ko`, `ja`) |
| `--task` | `transcribe` | 수행 작업: `transcribe` (전사) 또는 `translate` (번역) |

## 파일 구조

```text
stt/
├── prepare_model.py              # Step 1: 모델 폴더 준비
├── inference_mblt_model_zoo.py   # Step 2: 추론 실행
└── whisper-small-mxq/            # Step 1의 출력
    ├── config.json               # NPU 코어 할당이 포함된 모델 구성
    ├── whisper-small_encoder.mxq
    ├── whisper-small_decoder.mxq
    └── ...
```

## 참고 사항

- 오디오 파일은 자동으로 16kHz로 리샘플링됩니다
- 모델은 최대 30초 단위의 오디오 청크를 처리합니다
- Whisper는 99개 이상의 언어를 지원합니다. 주요 코드: `en`, `ko`, `ja`, `zh`, `es`, `fr`, `de`

## 참조

- [OpenAI Whisper](https://github.com/openai/whisper)
- [HuggingFace Whisper](https://huggingface.co/mobilint/whisper-small)
- [mblt-model-zoo Documentation](https://docs.mobilint.com/model-zoo)
- [Mobilint Documentation](https://docs.mobilint.com)
