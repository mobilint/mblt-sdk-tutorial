# 비전 언어 모델 (VLM) 런타임 추론

이 튜토리얼은 Aries 2 하드웨어에서 컴파일된 Qwen2-VL-2B-Instruct 모델로 추론을 실행하는 방법을 안내합니다.

이 튜토리얼에서는 [컴파일 튜토리얼](../../compilation/vlm/README.md)에서 컴파일한 [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) 모델을 사용합니다. **먼저 컴파일 튜토리얼을 완료해야** `compilation/vlm/mxq/`에 다음 파일들이 준비됩니다:

- `Qwen2-VL-2B-Instruct_text_model.mxq` - 컴파일된 언어 모델
- `Qwen2-VL-2B-Instruct_vision_transformer.mxq` - 컴파일된 비전 인코더
- `config.json` - MXQ 경로를 포함한 모델 구성 파일
- `model.safetensors` - 회전된 토큰 임베딩 가중치

## 개요

추론 과정은 두 단계로 구성됩니다:

1. **모델 준비**: MXQ 파일 복사 및 NPU 코어 할당 설정
2. **추론 실행**: mblt-model-zoo를 사용하여 이미지-텍스트-대-텍스트 추론 수행

이 튜토리얼은 [mblt-model-zoo](https://docs.mobilint.com/model-zoo)를 사용하여 간편한 추론 방식을 제공합니다. mblt-model-zoo를 통해 컴파일된 MXQ 모델을 한 줄(`AutoModelForImageTextToText.from_pretrained()`)로 로드할 수 있으며, 일반 HuggingFace 모델과 동일한 방식으로 사용할 수 있습니다. NPU 코어 할당, KV cache, 리소스 관리는 자동으로 처리됩니다.

모든 스크립트는 `runtime/vlm/` 디렉토리에서 실행합니다.

## 사전 요구 사항

```bash
pip install -r requirements.txt
```

## Step 1: 모델 폴더 준비

컴파일 출력을 복사하고 NPU 코어 할당을 설정합니다.

```bash
python prepare_model.py \
    --compilation-dir ../../compilation/vlm/mxq \
    --output-folder ./qwen2-vl-mxq \
    --model-id mobilint/Qwen2-VL-2B-Instruct
```

**수행 내용:**

- MXQ 파일, config.json, model.safetensors를 출력 폴더로 복사
- config.json에 NPU 코어 할당 설정(`target_cores`) 추가

**출력:**

- `./qwen2-vl-mxq/` - 추론 준비가 완료된 모델 폴더

## Step 2: mblt-model-zoo를 사용한 추론

이미지-텍스트-대-텍스트 추론을 실행합니다. 이 스크립트는 mblt-model-zoo의 공식 Qwen2-VL 구현을 사용하여 HuggingFace `AutoModel` API로 MXQ 모델을 로드합니다.

```bash
python inference_mblt_model_zoo.py \
    --model-folder ./qwen2-vl-mxq \
    --model-id mobilint/Qwen2-VL-2B-Instruct
```

**추가 옵션:**

```bash
# 로컬 이미지 사용
python inference_mblt_model_zoo.py --model-folder ./qwen2-vl-mxq --model-id mobilint/Qwen2-VL-2B-Instruct --image /path/to/image.jpg

# 커스텀 프롬프트
python inference_mblt_model_zoo.py --model-folder ./qwen2-vl-mxq --model-id mobilint/Qwen2-VL-2B-Instruct --prompt "이 이미지에는 무엇이 있나요?"

# 더 긴 생성
python inference_mblt_model_zoo.py --model-folder ./qwen2-vl-mxq --model-id mobilint/Qwen2-VL-2B-Instruct --max-length 1024
```

## NPU 추론 모드

NPU는 다양한 코어 모드를 지원합니다. 코어 모드는 `config.json`(`prepare_model.py`가 생성)에서 설정합니다.

| 모드 | 설명 | config.json 필드 |
|------|------|-----------------|
| `single` | 각 코어가 독립적으로 추론 실행. 기본 모드. | `target_cores: ["0:0"]` |
| `multi` | 여러 코어가 하나의 추론에 협력. | `core_mode: "multi"`, `target_clusters: [0]` |
| `global4` | 4개 코어(1 클러스터)가 글로벌 모드로 실행. | `core_mode: "global4"`, `target_clusters: [0]` |
| `global8` | 8개 코어(2 클러스터)가 글로벌 모드로 실행. 최대 처리량. | `core_mode: "global8"`, `target_clusters: [0, 1]` |

비전 인코더도 동일한 필드를 `vision_config` 아래에서 사용합니다 (예: `vision_config.target_cores`, `vision_config.core_mode`).

**예시: global8 모드로 변경**

`qwen2-vl-mxq/config.json`을 편집:
```json
{
    "core_mode": "global8",
    "target_clusters": [0, 1],
    "vision_config": {
        "core_mode": "global8",
        "target_clusters": [0, 1]
    }
}
```

> **참고:** `target_cores` 형식은 `"cluster:core"`입니다 (예: `"0:0"` = 클러스터 0, 코어 0). multi/global4/global8 사용 시에는 `target_cores` 대신 `target_clusters`를 사용합니다.

## 커맨드 라인 인자

### `prepare_model.py`

| 인자 | 기본값 | 설명 |
| --- | ----- | --- |
| `--compilation-dir` | `../../compilation/vlm/mxq` | 컴파일 출력 디렉토리 경로 |
| `--output-folder` | `./qwen2-vl-mxq` | 준비된 모델의 저장 폴더 |
| `--model-id` | `mobilint/Qwen2-VL-2B-Instruct` | mblt-model-zoo 모델 등록용 HuggingFace 모델 ID (config에 저장) |

### `inference_mblt_model_zoo.py`

| 인자 | 기본값 | 설명 |
| --- | ----- | --- |
| `--model-folder` | `./qwen2-vl-mxq` | 준비된 모델 폴더 경로 |
| `--model-id` | `mobilint/Qwen2-VL-2B-Instruct` | 프로세서(토크나이저 + 이미지 프로세서) 다운로드용 HuggingFace 모델 ID |
| `--image` | 데모 이미지 URL | 입력 이미지 경로 또는 URL |
| `--prompt` | `"Describe the environment..."` | 모델에 전달할 텍스트 프롬프트 |
| `--max-length` | `512` | 최대 생성 길이 |

## 파일 구조

```text
vlm/
├── prepare_model.py              # Step 1: 모델 폴더 준비
├── inference_mblt_model_zoo.py   # Step 2: 추론 실행
└── qwen2-vl-mxq/                # Step 1의 출력
    ├── config.json               # NPU 코어 할당이 포함된 모델 구성
    ├── model.safetensors
    ├── Qwen2-VL-2B-Instruct_text_model.mxq
    ├── Qwen2-VL-2B-Instruct_vision_transformer.mxq
    └── ...
```

## 문제 해결

### 모델을 찾을 수 없음 오류

컴파일 출력 디렉토리에 4개 파일이 모두 존재하는지 확인하세요. 없으면 컴파일 튜토리얼을 다시 실행하세요.

### Import 오류 (`No module named 'mblt_model_zoo'`)

```bash
pip install -r requirements.txt
```

### 메모리 부족 (OOM) 오류

- `--max-length`를 줄이세요
- 더 작은 이미지를 처리하세요
- NPU를 많이 사용하는 다른 애플리케이션을 종료하세요

## 참조

- [컴파일 튜토리얼](../../compilation/vlm/README.md)
- [Qwen2-VL 모델 카드](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [mblt-model-zoo Documentation](https://docs.mobilint.com/model-zoo)
- [Mobilint Documentation](https://docs.mobilint.com)
