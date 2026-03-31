import argparse

import librosa

# Register mblt-model-zoo's Whisper model with HuggingFace AutoModel.
# This single import enables AutoModelForSpeechSeq2Seq.from_pretrained()
# to load Mobilint MXQ models directly — no local wrapper needed.
import mblt_model_zoo.hf_transformers.models.whisper.modeling_whisper  # noqa: F401
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def main():
    parser = argparse.ArgumentParser(description="Whisper Speech-to-Text Inference using mblt-model-zoo")
    parser.add_argument(
        "--audio",
        type=str,
        default="../../compilation/stt/audio_files/en_us_0000.wav",
        help="Path to the audio file to transcribe",
    )
    parser.add_argument(
        "--model-folder",
        type=str,
        default="./whisper-small-mxq",
        help="Path to the folder containing compiled MXQ models",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="mobilint/whisper-small",
        help="HuggingFace model ID for processor download",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Source language code (e.g., 'en', 'ko', 'ja'). If not specified, auto-detect.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Task: 'transcribe' or 'translate' (to English)",
    )

    args = parser.parse_args()

    # Load model from config.json in model folder.
    # config.json contains MXQ paths and NPU core allocation (target_cores).
    # Core mode can be changed by editing config.json — see README for details.
    print(f"Loading model from {args.model_folder}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_folder)

    # Load processor from HuggingFace
    processor = AutoProcessor.from_pretrained(args.model_id)

    # Load and preprocess audio (resample to 16kHz)
    print(f"Loading audio: {args.audio}")
    audio_array, _ = librosa.load(args.audio, sr=16000)
    input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features

    # Prepare generation kwargs
    generate_kwargs = {"max_new_tokens": 444}
    if args.language:
        generate_kwargs["language"] = args.language
    if args.task:
        generate_kwargs["task"] = args.task

    print(f"Running inference... (task: {args.task}, language: {args.language or 'auto-detect'})")

    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features, **generate_kwargs)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # Clean up NPU resources
    model.model.encoder.dispose()
    model.model.decoder.dispose()

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(transcription)
    print("=" * 60)


if __name__ == "__main__":
    main()
