from argparse import ArgumentParser
from pathlib import Path

import mblt_model_zoo.hf_transformers.models.whisper.modeling_whisper  # noqa: F401
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUDIO = REPO_ROOT / "compilation/stt/audio_files/English/en_us_0000.wav"
DEFAULT_MODEL_FOLDER = REPO_ROOT / "compilation/stt/prepared/aries-rb/whisper-small"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL_FOLDER)
    parser.add_argument("--language")
    parser.add_argument("--task", choices=("transcribe", "translate"), default="transcribe")
    args = parser.parse_args()

    audio, sample_rate = sf.read(args.audio, dtype="float32")
    if sample_rate != 16000:
        raise ValueError(f"Whisper requires 16 kHz audio, got {sample_rate} Hz: {args.audio}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    model_folder = str(args.model_folder)
    processor = AutoProcessor.from_pretrained(model_folder, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_folder)
    input_features = processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_features
    generation_options = {"task": args.task}
    if args.language:
        generation_options["language"] = args.language

    try:
        with torch.inference_mode():
            predicted_ids = model.generate(input_features, **generation_options)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    finally:
        model.model.encoder.dispose()
        model.model.decoder.dispose()

    print(transcription)
