# VLM 런타임

이 튜토리얼은 Mobilint NPU 하드웨어에서 컴파일된 `Qwen3-VL-2B-Instruct` 모델을 실행하는 방법을 설명합니다.

시작하기 전에 [../../../compilation/vlm/README.KR.md](../../../compilation/vlm/README.KR.md)의 컴파일 과정을 먼저 완료하세요. 이 디렉토리의 런타임 예제는 `../../../compilation/vlm/mxq/`에 다음 파일이 준비되어 있다고 가정합니다.

- `Qwen3-VL-2B-Instruct_text_model.mxq`
- `Qwen3-VL-2B-Instruct_vision_transformer.mxq`
- `config.json`
- `model.safetensors`

## 사전 준비

필요한 패키지를 설치하세요.

```bash
pip install -r requirements.txt
```

## 개요

이 튜토리얼은 `mblt-model-zoo`를 사용해 Hugging Face 스타일 API로 멀티모달 이미지-텍스트 추론을 실행합니다. 런타임 흐름은 두 단계로 구성됩니다.

1. 컴파일 결과로부터 모델 폴더를 준비합니다.
2. 이미지와 프롬프트를 사용해 이미지-텍스트 생성을 실행합니다.

준비된 모델 폴더에는 텍스트 모델 MXQ, 비전 인코더 MXQ, 설정 파일, safetensors 가중치, NPU 코어 할당 정보가 함께 저장됩니다.

## 이 튜토리얼의 파일

- `prepare_model.py`: 컴파일 결과로부터 준비된 모델 폴더를 생성합니다.
- `inference_mblt_model_zoo.py`: 이미지-텍스트-대-텍스트 생성을 실행합니다.
- `requirements.txt`: 이 튜토리얼에 필요한 Python 의존성입니다.

## Step 1: 모델 폴더 준비

```bash
python prepare_model.py \
    --compilation-dir ../../../compilation/vlm/mxq \
    --output-folder ./qwen3-vl-mxq \
    --model-id mobilint/Qwen3-VL-2B-Instruct
```

이 스크립트는 다음 작업을 수행합니다.

- 컴파일된 MXQ 파일, `config.json`, `model.safetensors` 복사
- `config.json`에 기본 NPU 코어 할당 설정 추가
- 준비된 폴더가 사용할 모델 ID에 맞게 `_name_or_path` 업데이트

## Step 2: 추론 실행

기본 예제를 실행하려면 다음 명령을 사용하세요.

```bash
python inference_mblt_model_zoo.py \
    --model-folder ./qwen3-vl-mxq \
    --model-id mobilint/Qwen3-VL-2B-Instruct
```

자주 쓰는 옵션:

```bash
python inference_mblt_model_zoo.py --model-folder ./qwen3-vl-mxq --model-id mobilint/Qwen3-VL-2B-Instruct --image /path/to/image.jpg
python inference_mblt_model_zoo.py --model-folder ./qwen3-vl-mxq --model-id mobilint/Qwen3-VL-2B-Instruct --prompt "이 이미지에는 어떤 물체가 있나요?"
python inference_mblt_model_zoo.py --model-folder ./qwen3-vl-mxq --model-id mobilint/Qwen3-VL-2B-Instruct --max-length 1024
```

이 스크립트는 `image-text-to-text` 파이프라인을 구성하고, 이미지와 프롬프트를 함께 모델에 전달한 뒤 생성 결과를 스트리밍 출력합니다.

## NPU 코어 모드

생성된 `config.json`을 편집해 언어 모델과 비전 인코더의 코어 할당을 바꿀 수 있습니다.

| 모드 | 설명 | 예시 언어 모델 필드 |
| --- | --- | --- |
| `single` | 단일 코어 실행 | `target_cores: ["0:0"]` |
| `multi` | 여러 코어가 하나의 추론에 협력 | `core_mode: "multi"`, `target_clusters: [0]` |
| `global4` | 한 클러스터를 글로벌 모드로 사용 | `core_mode: "global4"`, `target_clusters: [0]` |
| `global8` | 두 클러스터를 글로벌 모드로 사용 | `core_mode: "global8"`, `target_clusters: [0, 1]` |

비전 인코더는 `vision_config` 아래에서 같은 패턴을 사용합니다.

## 파라미터

### `prepare_model.py`

- `--compilation-dir`: 컴파일 출력 디렉토리 경로
- `--output-folder`: 준비된 모델 저장 폴더
- `--model-id`: 준비된 설정 파일에 저장할 Hugging Face 모델 ID

### `inference_mblt_model_zoo.py`

- `--model-folder`: 준비된 모델 폴더 경로
- `--model-id`: 프로세서 다운로드에 사용할 Hugging Face 모델 ID. 컴파일된 `config.json`과 동일한 Qwen3 VLM 런타임 등록값을 사용하세요.
- `--image`: 입력 이미지의 로컬 경로 또는 URL
- `--prompt`: 이미지와 함께 전달할 프롬프트 텍스트
- `--max-length`: 최대 생성 길이

## 예상 출력

스크립트는 입력 이미지에 대해 설명하거나 질문에 답하는 생성 텍스트를 스트리밍 출력합니다.
