# Speech-to-Text Model Compilation

This tutorial compiles the encoder and decoder of [OpenAI Whisper Small](https://huggingface.co/openai/whisper-small) into MXQ models and prepares one self-contained runtime model directory.

Run all commands from `compilation/stt`.

## Prerequisites

```bash
pip install -r requirements.txt
```

The Whisper parser requires `transformers==4.50.0`. The version is pinned in `requirements.txt`.

## Supported Devices

| Device | Support |
| --- | --- |
| `aries-rb` | Supported |
| `regulus-rb` | Supported |
| `regulus-ra` | Not supported |

## 1. Prepare Audio Data

```bash
python prepare_audio.py
```

The script downloads 20 validation samples for each of 17 FLEURS languages and saves 340 pairs of 16 kHz PCM WAV files and transcripts under `./audio_files`.

```text
audio_files/
├── English/
│   ├── en_us_0000.wav
│   └── en_us_0000.txt
├── Korean/
└── ...
```

## 2. Generate Calibration Data

```bash
python generate_calibration.py
```

The encoder receives one log-mel spectrogram per audio file. The decoder uses five audio-length fractions, restores the Whisper language/task prefix, removes terminal EOS, and excludes samples shorter than eight tokens.

The output is written to `./calibration_data`.

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

The output directory must be empty before generation.

## 3. Compile MXQ Models

For ARIES:

```bash
python compile_encoder.py --target-device aries-rb
python compile_decoder.py --target-device aries-rb
```

For REGULUS:

```bash
python compile_encoder.py --target-device regulus-rb
python compile_decoder.py --target-device regulus-rb
```

The scripts always rebuild the target-specific MBLT before compiling MXQ.

```text
mblt/<target-device>/whisper-small_{encoder,decoder}.mblt
mxq/<target-device>/whisper-small_{encoder,decoder}.mxq
```

The compiler configuration from the validated Whisper experiments is applied automatically.

- ARIES uses `inference_scheme="all"`; REGULUS uses `inference_scheme="single"`.
- Encoder and decoder use the validated equivalent transformations and mixed-precision activation configuration.
- Decoder additionally uses max calibration, full-sequence LLM calibration, and Hessian quantization.
- REGULUS decoder sequence and cache lengths are limited to 1024.

## 4. Prepare the Runtime Model

Run this after both MXQ files have been compiled.

```bash
python prepare_model.py --target-device aries-rb
```

For REGULUS:

```bash
python prepare_model.py --target-device regulus-rb
```

The output is written to `./prepared/<target-device>/whisper-small`. It contains the processor, tokenizer, configuration, CPU embedding weights, and both compiled MXQ files required by the runtime.

If the output directory already exists, pass `--force` to replace it.

## Runtime

Continue with the [Python STT runtime tutorial](../../runtime/python/stt/README.md).
