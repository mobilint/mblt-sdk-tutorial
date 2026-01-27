# 음성-텍스트 변환(STT) 모델 컴파일

이 튜토리얼은 Mobilint qbcompiler 컴파일러를 사용하여 Whisper 음성-텍스트 변환 모델을 컴파일하는 방법을 상세히 설명합니다. 컴파일 과정에서는 Whisper 모델(인코더와 디코더를 별도로)을 Mobilint NPU 하드웨어에서 효율적으로 실행할 수 있는 최적화된 `.mxq` 형식으로 변환합니다.

이 튜토리얼에서는 OpenAI가 개발한 다국어 음성 인식 모델인 [Whisper Small](https://huggingface.co/openai/whisper-small) 모델을 사용합니다.

## 사전 요구사항

시작하기 전에 다음 사항을 확인하세요:

- Python 3.8 이상
- CUDA를 지원하는 GPU (캘리브레이션 및 컴파일에 필요)
- 충분한 디스크 공간 (모델 + 캘리브레이션 데이터를 위해 약 10GB)

### qbcompiler 컴파일러 설치

Mobilint 릴리즈 페이지에서 qbcompiler 컴파일러(버전 0.12.0.0)를 다운로드하여 설치하세요:

1. [https://dl.mobilint.com/releases?series-id=1](https://dl.mobilint.com/releases?series-id=1)로 이동
2. qbcompiler 버전 **0.12.0.0**에 해당하는 `.whl` 파일 다운로드
3. wheel 파일 설치:

```bash
pip install qbcompiler-0.12.0.0-<your-platform>.whl
```

## 개요

컴파일 과정은 세 가지 주요 단계로 구성됩니다:

1. **데이터 다운로드**: FLEURS 데이터셋에서 다국어 오디오 데이터 다운로드
2. **캘리브레이션 데이터 생성**: 인코더와 디코더용 캘리브레이션 데이터셋 생성
3. **모델 컴파일**: 인코더와 디코더를 각각 `.mxq` 형식으로 컴파일

## 1단계: 데이터 다운로드

먼저 Google FLEURS 데이터셋에서 오디오 데이터를 다운로드합니다. 이 데이터에는 캘리브레이션에 사용될 다국어 오디오 샘플이 포함되어 있습니다.

### 의존성 설치

```bash
cd data
```

**필요한 패키지:**
- datasets==3.6.0
- librosa
- soundfile
- openai-whisper

### 오디오 데이터 다운로드

```bash
python download_data.py
```

**실행 내용:**
- 17개 언어에 대한 Google/FLEURS 데이터셋의 오디오 샘플 다운로드
- 전사 및 번역 생성을 위해 Whisper Large V3 로드
- 오디오 파일을 16kHz 모노 WAV 형식으로 리샘플링하여 저장
- 원본 언어를 지정하여 Whisper로 전사 생성
- Whisper의 번역 기능을 사용하여 영어 번역 생성

**지원 언어:**
- 아랍어, 중국어(표준어), 독일어, 그리스어, 영어, 스페인어, 프랑스어
- 인도네시아어, 이탈리아어, 일본어, 한국어, 포르투갈어, 러시아어
- 타밀어, 태국어, 우르두어, 베트남어

**출력:**
- `audio_files/` - WAV 오디오 파일이 포함된 디렉토리
- `transcriptions.json` - 각 오디오 파일에 대한 Whisper 전사
- `translations.json` - 메타데이터가 포함된 영어 번역

## 2단계: 캘리브레이션 데이터 생성

Whisper 인코더와 디코더 모두에 대한 캘리브레이션 데이터를 생성합니다. 이 데이터는 컴파일 중 양자화에 필수적입니다.

### 의존성 설치

```bash
cd ../calibration
```

**필요한 패키지:**
- torch==2.7.1
- transformers==4.50.0
- librosa

### 캘리브레이션 데이터 생성

```bash
python create_calibration.py
```

**실행 내용:**
- Whisper 인코더용 캘리브레이션 데이터 생성 (멜 스펙트로그램 특징)
- Whisper 디코더용 캘리브레이션 데이터 생성 (인코더 은닉 상태 + 디코더 임베딩)
- whisper-small을 사용하여 전사 및 번역을 즉석에서 생성
- 적절한 특수 토큰과 함께 전사 및 번역 작업 지원
- 다양한 캘리브레이션을 위해 전사(80%)와 번역(20%) 작업을 무작위로 혼합

**인코더 캘리브레이션:**
- HuggingFace Whisper 프로세서를 통해 오디오 파일 처리
- `[1, 80, 3000]` 형태의 멜 스펙트로그램 특징 추출 (저장을 위해 `[1, 3000, 80]`으로 전치)
- `.npy` 형식으로 캘리브레이션 파일 저장

**디코더 캘리브레이션:**
- 은닉 상태를 얻기 위해 인코더를 통해 오디오 처리
- whisper-small 모델을 사용하여 전사 및 번역 생성
- HuggingFace 프로세서를 사용하여 텍스트 정규화 적용
- 적절한 특수 토큰(SOT, 언어, 작업, 타임스탬프)이 포함된 토큰 시퀀스 생성
- 디코더 캘리브레이션을 위한 입력 임베딩 생성 (토큰 임베딩 + 위치 임베딩)
- 인코더와 디코더 은닉 상태 모두 저장

**출력:**
- `encoder/` - 인코더 캘리브레이션 데이터
  - `whisper_encoder_cali.txt` - 캘리브레이션 파일 경로 목록
  - `encoder_calib_*.npy` - 개별 캘리브레이션 샘플
- `decoder/` - 디코더 캘리브레이션 데이터
  - `whisper_decoder_calib.json` - 캘리브레이션 메타데이터 및 경로
  - `whisper_decoder_calib_metadata.json` - 참조용 디코딩된 토큰 시퀀스
  - `sample_*/encoder_hidden_states.npy` - 인코더 출력
  - `sample_*/decoder_hidden_states.npy` - 디코더 입력

## 3단계: 모델 컴파일

캘리브레이션 데이터를 사용하여 인코더와 디코더를 `.mxq` 형식으로 컴파일합니다.

### 의존성 설치

```bash
cd ../compilation
```

**필요한 패키지:**
- transformers==4.50.0
- qbcompiler==0.12.0.0 (사전 요구사항에 설명된 수정 포함)

### 인코더 컴파일

```bash
python compile_encoder.py
```

**실행 내용:**
- HuggingFace에서 Whisper Small 모델 로드
- 먼저 인코더를 MBLT 형식으로 컴파일
- 양자화를 위해 인코더 캘리브레이션 데이터 사용
- `global4` 추론 방식을 사용하여 최종 `.mxq` 형식으로 컴파일

**출력:**
- `compiled/whisper-small_encoder.mblt` - 중간 MBLT 형식
- `compiled/whisper-small_encoder.mxq` - NPU용 최종 양자화 모델

### 디코더 컴파일

> **중요:** 디코더 컴파일을 실행하기 전에 사전 요구사항에 설명된 대로 qbcompiler의 parser.py 수정을 적용했는지 확인하세요.

```bash
python compile_decoder.py
```

**실행 내용:**
- HuggingFace에서 Whisper Small 모델 로드
- 먼저 디코더를 MBLT 형식으로 컴파일
- 전체 시퀀스 길이 캘리브레이션과 함께 디코더 캘리브레이션 데이터 사용 (`use_full_seq_len_calib=True`)
- 디코더 컴파일을 위해 `get_llm_config()`를 통한 LLM 특화 구성 적용
- 최종 `.mxq` 형식으로 컴파일

**출력:**
- `compiled/whisper-small_decoder.mblt` - 중간 MBLT 형식
- `compiled/whisper-small_decoder.mxq` - NPU용 최종 양자화 모델

## 전체 컴파일 파이프라인

전체 명령어 순서는 다음과 같습니다:

```bash
# 1단계: 오디오 데이터 다운로드
cd data
python download_data.py

# 2단계: 캘리브레이션 데이터 생성
cd ../calibration
python create_calibration.py

# 3단계: 모델 컴파일
cd ../compilation

# 인코더 컴파일
python compile_encoder.py

# 디코더 컴파일 (qbcompiler 수정 필요 - 사전 요구사항 참조)
python compile_decoder.py
```

## 출력 요약

모든 단계를 완료하면 다음 파일들이 생성됩니다:

### 데이터 파일
- `data/audio_files/` - FLEURS 데이터셋의 오디오 샘플
- `data/transcriptions.json` - Whisper 전사
- `data/translations.json` - 영어 번역

### 캘리브레이션 파일
- `calibration/encoder/` - 인코더 캘리브레이션 데이터
- `calibration/decoder/` - 디코더 캘리브레이션 데이터

### 컴파일된 모델
- `compilation/compiled/whisper-small_encoder.mxq` - NPU용 양자화된 인코더
- `compilation/compiled/whisper-small_decoder.mxq` - NPU용 양자화된 디코더

## 문제 해결

### 메모리 부족 오류

- 충분한 VRAM을 가진 GPU가 있는지 확인 (8GB 이상 권장)
- 다른 GPU 집약적인 애플리케이션 종료
- 필요한 경우 캘리브레이션 샘플 수 감소

### 캘리브레이션 데이터 누락

캘리브레이션 데이터 누락으로 컴파일이 실패하는 경우:

```bash
# 캘리브레이션 파일 존재 확인
ls ../calibration/encoder/whisper_encoder_cali.txt
ls ../calibration/decoder/whisper_decoder_calib.json
```

파일이 없으면 calibration 폴더에서 `create_calibration.py`를 다시 실행하세요.

### 오디오 다운로드 문제

- FLEURS 데이터셋 다운로드를 위한 안정적인 인터넷 연결 확인
- 다운로드 스크립트는 HuggingFace 데이터셋에 대한 접근이 필요합니다

## 파일 구조

```
stt/
├── README.md
├── data/
│   ├── download_data.py
│   ├── audio_files/                        # 다운로드된 오디오 샘플
│   ├── transcriptions.json                 # 생성된 전사
│   └── translations.json                   # 생성된 번역
├── calibration/
│   ├── create_calibration.py
│   ├── encoder/                            # 인코더 캘리브레이션 데이터
│   │   ├── whisper_encoder_cali.txt        # 캘리브레이션 파일 목록
│   │   └── encoder_calib_*.npy             # 멜 스펙트로그램 특징
│   └── decoder/                            # 디코더 캘리브레이션 데이터
│       ├── whisper_decoder_calib.json      # 캘리브레이션 설정 및 경로
│       ├── whisper_decoder_calib_metadata.json  # 디코딩된 토큰
│       └── sample_*/                       # 샘플별 데이터
│           ├── encoder_hidden_states.npy
│           └── decoder_hidden_states.npy
└── compilation/
    ├── compile_encoder.py
    ├── compile_decoder.py
    └── compiled/                           # 출력 MXQ 모델
        ├── whisper-small_encoder.mblt
        ├── whisper-small_encoder.mxq
        ├── whisper-small_decoder.mblt
        └── whisper-small_decoder.mxq
```

## 참고 자료

- [OpenAI Whisper](https://github.com/openai/whisper)
- [HuggingFace Whisper](https://huggingface.co/openai/whisper-small)
- [Google FLEURS 데이터셋](https://huggingface.co/datasets/google/fleurs)
- [Mobilint 문서](https://docs.mobilint.com)

## 지원

문제나 질문이 있는 경우:
- 위의 문제 해결 섹션 확인
- qbcompiler SDK 문서 검토
- 상세한 오류 로그와 함께 Mobilint 지원팀에 문의
