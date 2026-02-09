#!/usr/bin/env python3
"""
Whisper Speech-to-Text Inference using Mobilint MXQ Models

This script demonstrates how to run inference on compiled Whisper MXQ models
using the local mblt-whisper module.
"""

import argparse
import os
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader

import librosa
import torch

# Load mblt-whisper.py module to register HuggingFace Auto classes
_current_dir = os.path.dirname(os.path.abspath(__file__))
_spec = spec_from_loader(
    "mblt_whisper",
    SourceFileLoader("mblt_whisper", os.path.join(_current_dir, "mblt-whisper.py")),
)
mblt_whisper = module_from_spec(_spec)
_spec.loader.exec_module(mblt_whisper)

# Now Auto classes are registered and can be used
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def transcribe_audio(
    audio_path: str,
    model_folder: str,
    language: str = None,
    task: str = "transcribe",
):
    """
    Transcribe or translate audio using compiled Whisper MXQ model.

    Args:
        audio_path: Path to the audio file
        model_folder: Path to the folder containing compiled MXQ models
        language: Source language code (e.g., 'en', 'ko', 'ja'). If None, auto-detect.
        task: 'transcribe' for transcription, 'translate' for translation to English

    Returns:
        Transcription or translation text
    """
    # Load model from compiled MXQ files (Auto classes registered by mblt-whisper)
    print(f"Loading model from {model_folder}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_folder)

    # Load processor (tokenizer + feature extractor)
    processor = AutoProcessor.from_pretrained(model_folder)

    # Create pipeline for automatic speech recognition
    from transformers import pipeline

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=1,
    )

    # Prepare generation kwargs
    generate_kwargs = {}
    if language:
        generate_kwargs["language"] = language
    if task:
        generate_kwargs["task"] = task

    print(f"Processing audio: {audio_path}")
    print(f"Task: {task}, Language: {language if language else 'auto-detect'}")

    # Run inference
    result = pipe(audio_path, generate_kwargs=generate_kwargs)

    # Clean up resources
    pipe.model.dispose()

    return result["text"]


def transcribe_audio_manual(
    audio_path: str,
    model_folder: str,
    language: str = None,
    task: str = "transcribe",
):
    """
    Transcribe audio using manual model invocation (without pipeline).
    This gives more control over the inference process.

    Args:
        audio_path: Path to the audio file
        model_folder: Path to the folder containing compiled MXQ models
        language: Source language code (e.g., 'en', 'ko', 'ja'). If None, auto-detect.
        task: 'transcribe' for transcription, 'translate' for translation to English

    Returns:
        Transcription or translation text
    """
    # Load model from compiled MXQ files (Auto classes registered by mblt-whisper)
    print(f"Loading model from {model_folder}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_folder)

    # Load processor (tokenizer + feature extractor)
    processor = AutoProcessor.from_pretrained(model_folder)

    # Load and preprocess audio
    print(f"Loading audio: {audio_path}")
    audio_array, sampling_rate = librosa.load(audio_path, sr=16000)

    # Process audio through feature extractor
    input_features = processor(
        audio_array, sampling_rate=16000, return_tensors="pt"
    ).input_features

    # Prepare generation kwargs
    generate_kwargs = {
        "max_new_tokens": 444,
    }
    if language:
        generate_kwargs["language"] = language
    if task:
        generate_kwargs["task"] = task
    print(f"Running inference...")
    print(f"Task: {task}, Language: {language if language else 'auto-detect'}")

    # Generate transcription
    print(input_features.shape, flush=True)
    with torch.no_grad():
        predicted_ids = model.generate(input_features, **generate_kwargs)

    # Decode tokens to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # Clean up resources
    model.dispose()

    return transcription


def main():
    parser = argparse.ArgumentParser(
        description="Whisper Speech-to-Text Inference using Mobilint MXQ Models"
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="../../../compilation/transformers/stt/data/audio_files/en_us_0000.wav",
        help="Path to the audio file to transcribe",
    )
    parser.add_argument(
        "--model-folder",
        type=str,
        default="./whisper-small-mxq",
        help="Path to the folder containing compiled MXQ models",
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
        help="Task: 'transcribe' for transcription, 'translate' for translation to English",
    )
    parser.add_argument(
        "--use-pipeline",
        action="store_true",
        help="Use pipeline API instead of manual inference",
    )

    args = parser.parse_args()

    if args.use_pipeline:
        result = transcribe_audio(
            audio_path=args.audio,
            model_folder=args.model_folder,
            language=args.language,
            task=args.task,
        )
    else:
        result = transcribe_audio_manual(
            audio_path=args.audio,
            model_folder=args.model_folder,
            language=args.language,
            task=args.task,
        )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()
