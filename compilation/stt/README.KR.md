# 음성-텍스트 변환(STT) 모델 컴파일

이 튜토리얼은 Mobilint qbcompiler 컴파일러를 사용하여 Whisper 음성-텍스트 변환 모델을 컴파일하는 방법을 상세히 설명합니다. 컴파일 과정에서는 Whisper 모델(인코더와 디코더를 별도로)을 Mobilint NPU 하드웨어에서 효율적으로 실행할 수 있는 최적화된 `.mxq` 형식으로 변환합니다.

이 튜토리얼에서는 OpenAI가 개발한 다국어 음성 인식 모델인 [Whisper Small](https://huggingface.co/openai/whisper-small) 모델을 사용합니다.

## 개요

컴파일 과정은 세 가지 주요 단계로 구성됩니다:

1. **데이터 준비**: FLEURS 데이터셋에서 다국어 오디오 데이터 다운로드
2. **캘리브레이션 데이터 생성**: 인코더와 디코더용 캘리브레이션 데이터셋 생성
3. **모델 컴파일**: 인코더와 디코더를 각각 `.mxq` 형식으로 컴파일

모든 스크립트는 `stt/` 디렉토리에서 실행합니다.

## 사전 준비

```bash
pip install -r requirements.txt
```

## 1단계: 오디오 데이터 준비

Google FLEURS 데이터셋에서 오디오 데이터를 다운로드합니다. 이 데이터에는 캘리브레이션에 사용될 다국어 오디오 샘플이 포함되어 있습니다.

```bash
python prepare_audio.py
```

**실행 내용:**

- 17개 언어에 대한 Google/FLEURS 데이터셋의 오디오 샘플 다운로드
- 오디오 파일을 16kHz 모노 WAV 형식으로 리샘플링하여 저장

**지원 언어:**

- 아랍어, 중국어(표준어), 독일어, 그리스어, 영어, 스페인어, 프랑스어
- 인도네시아어, 이탈리아어, 일본어, 한국어, 포르투갈어, 러시아어
- 타밀어, 태국어, 우르두어, 베트남어

**출력:**

- `./audio_files/` - WAV 오디오 파일이 포함된 디렉토리

## 2단계: 캘리브레이션 데이터 생성

Whisper 인코더와 디코더 모두에 대한 캘리브레이션 데이터를 생성합니다. 이 데이터는 컴파일 중 양자화에 필수적입니다.

이 단계에서는 내부적으로 **Whisper Small** 모델을 로드하여 현실적인 캘리브레이션 입력을 생성합니다. 인코더 캘리브레이션은 오디오에서 멜 스펙트로그램을 추출하고, 디코더 캘리브레이션은 모델을 사용하여 전사/번역을 생성한 뒤 토큰 임베딩으로 변환합니다.

> **참고:** 이 단계에서는 디코더 캘리브레이션 데이터를 생성하기 위해 Whisper Small 모델로 추론을 수행합니다. GPU(CUDA)가 감지되면 자동으로 사용되어 데이터 생성 속도가 크게 향상됩니다. CPU에서도 정상 동작하지만 처리 시간이 더 오래 걸립니다.

```bash
python generate_calibration.py
```

**실행 내용:**

- Whisper 인코더용 캘리브레이션 데이터 생성 (멜 스펙트로그램 특징)
- Whisper 디코더용 캘리브레이션 데이터 생성 (인코더 은닉 상태 + 디코더 임베딩)
- Whisper Small 모델을 로드하여 전사 및 번역을 즉석에서 생성
- 다양한 캘리브레이션을 위해 전사(80%)와 번역(20%) 작업을 무작위로 혼합

**출력:**

- `./calibration_data/encoder/` - 인코더 캘리브레이션 데이터
  - `whisper_encoder_cali.txt` - 캘리브레이션 파일 경로 목록
  - `encoder_calib_*.npy` - 개별 캘리브레이션 샘플
- `./calibration_data/decoder/` - 디코더 캘리브레이션 데이터
  - `whisper_decoder_calib.json` - 캘리브레이션 메타데이터 및 경로
  - `sample_*/encoder_hidden_states.npy` - 인코더 출력
  - `sample_*/decoder_hidden_states.npy` - 디코더 입력

## 3단계: 모델 컴파일

캘리브레이션 데이터를 사용하여 인코더와 디코더를 `.mxq` 형식으로 컴파일합니다.

### 인코더 컴파일

```bash
python compile_encoder.py
```

- HuggingFace에서 Whisper Small 모델 로드
- 인코더를 MBLT 형식으로 컴파일 후 `all` 추론 방식으로 `.mxq` 변환

**출력:**

- `./mblt/whisper-small_encoder.mblt` - 중간 MBLT 형식
- `./mxq/whisper-small_encoder.mxq` - NPU용 최종 양자화 모델

### 디코더 컴파일

```bash
python compile_decoder.py
```

- HuggingFace에서 Whisper Small 모델 로드
- 디코더를 MBLT 형식으로 컴파일 후 `LlmConfig`로 `.mxq` 변환

**출력:**

- `./mblt/whisper-small_decoder.mblt` - 중간 MBLT 형식
- `./mxq/whisper-small_decoder.mxq` - NPU용 최종 양자화 모델

## 문제 해결

### 메모리 부족 오류

- GPU 사용 시 충분한 VRAM 확인 (8GB 이상 권장)
- 다른 GPU 집약적인 애플리케이션 종료
- 필요한 경우 캘리브레이션 샘플 수 감소
- 또는 CPU에서 캘리브레이션 데이터 생성 실행 (자동 폴백)

### 캘리브레이션 데이터 누락

캘리브레이션 데이터 누락으로 컴파일이 실패하는 경우:

```bash
ls ./calibration_data/encoder/whisper_encoder_cali.txt
ls ./calibration_data/decoder/whisper_decoder_calib.json
```

파일이 없으면 `generate_calibration.py`를 다시 실행하세요.

### 오디오 다운로드 문제

- FLEURS 데이터셋 다운로드를 위한 안정적인 인터넷 연결 확인
- 다운로드 스크립트는 HuggingFace 데이터셋에 대한 접근이 필요합니다

## 파일 구조

```text
stt/
├── prepare_audio.py
├── generate_calibration.py
├── compile_encoder.py
├── compile_decoder.py
├── README.md
├── README.KR.md
├── audio_files/                            # 다운로드된 오디오 샘플
├── calibration_data/                       # 캘리브레이션 데이터
│   ├── encoder/
│   │   ├── whisper_encoder_cali.txt
│   │   └── encoder_calib_*.npy
│   └── decoder/
│       ├── whisper_decoder_calib.json
│       └── sample_*/
│           ├── encoder_hidden_states.npy
│           └── decoder_hidden_states.npy
├── mblt/                                   # 중간 MBLT 모델
│   ├── whisper-small_encoder.mblt
│   └── whisper-small_decoder.mblt
└── mxq/                                    # 출력 MXQ 모델
    ├── whisper-small_encoder.mxq
    └── whisper-small_decoder.mxq
```

## 참고 자료

- [OpenAI Whisper](https://github.com/openai/whisper)
- [HuggingFace Whisper](https://huggingface.co/openai/whisper-small)
- [Google FLEURS 데이터셋](https://huggingface.co/datasets/google/fleurs)
- [Mobilint 문서](https://docs.mobilint.com)
