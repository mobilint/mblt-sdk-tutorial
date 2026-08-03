"""Generate calibration data for the Whisper encoder and decoder.

Two calibration sets are built from the FLEURS audio in ./audio_files:

  Encoder: each clip's log-mel spectrogram ([1, 3000, 80]) is saved as one .npy.
  Decoder: each clip is transcribed/translated on-the-fly and its decoder input
           embeddings (token + positional) are saved. Every clip is cut to several
           length prefixes (FRACTIONS_TO_USE) so the decoder sees diverse sequence lengths.

Output:
  calibration_data/encoder/encoder_calib_NNNN.npy
  calibration_data/encoder/whisper_encoder_cali.txt              # list qbcompiler reads
  calibration_data/decoder/sample_NNNN/{encoder,decoder}_hidden_states.npy
  calibration_data/decoder/whisper_decoder_calib.json            # info + calib paths
"""

import json
import os
import random

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE_MODEL = "openai/whisper-small"

# FLEURS filename prefix -> Whisper language code (used to condition decoder generation).
FLEURS_TO_WHISPER = {
    "ar_eg": "ar",
    "cmn_hans_cn": "zh",
    "de_de": "de",
    "el_gr": "el",
    "en_us": "en",
    "es_419": "es",
    "fr_fr": "fr",
    "id_id": "id",
    "it_it": "it",
    "ja_jp": "ja",
    "ko_kr": "ko",
    "pt_br": "pt",
    "ru_ru": "ru",
    "ta_in": "ta",
    "th_th": "th",
    "ur_pk": "ur",
    "vi_vn": "vi",
}

# Length prefixes (fraction of each clip) for decoder calibration, so the decoder sees diverse
FRACTIONS_TO_USE = [0.2, 0.4, 0.6, 0.8, 1.0]


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs so file selection and the translate/transcribe split are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_wavs(audio_dir, num_samples):
    """Return up to num_samples wav filenames from audio_dir, sorted for reproducibility."""
    return sorted(f for f in os.listdir(audio_dir) if f.endswith(".wav"))[:num_samples]


def detect_language(filename):
    """Map a FLEURS-prefixed filename (e.g. 'ko_kr_0001.wav') to a Whisper language code, or None."""
    for prefix, code in FLEURS_TO_WHISPER.items():
        if filename.startswith(prefix):
            return code
    return None


def save_sample(output_dir, idx, encoder_hidden, decoder_hidden):
    """Save one (encoder, decoder) hidden-state pair under sample_{idx:04d}/ and return their paths."""
    sample_dir = os.path.join(os.path.abspath(output_dir), f"sample_{idx:04d}")
    os.makedirs(sample_dir, exist_ok=True)
    encoder_path = os.path.join(sample_dir, "encoder_hidden_states.npy")
    decoder_path = os.path.join(sample_dir, "decoder_hidden_states.npy")
    np.save(encoder_path, encoder_hidden)
    np.save(decoder_path, decoder_hidden)
    return encoder_path, decoder_path


def generate_encoder_calibration_data(processor, audio_dir, output_dir="./calibration_data/encoder", num_samples=1000):
    """Save each clip's mel spectrogram ([1, 3000, 80]) as encoder calibration input."""
    print("Generating encoder calibration data...")
    os.makedirs(output_dir, exist_ok=True)

    wavs = list_wavs(audio_dir, num_samples)
    print(f"Processing {len(wavs)} audio files...")

    # ===== Build mel-spectrogram samples =====
    calib_paths = []
    for i, wav in enumerate(tqdm(wavs, desc="Encoder calibration", unit="file")):
        try:
            audio, _ = librosa.load(os.path.join(audio_dir, wav), sr=16000)
            mel = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
            # mel spectrogram: [1, 80, 3000] -> [1, 3000, 80]
            mel = np.ascontiguousarray(mel.transpose(1, 2).cpu().numpy().astype(np.float32))
            path = os.path.join(output_dir, f"encoder_calib_{i:04d}.npy")
            np.save(path, mel)
            calib_paths.append(os.path.abspath(path))
        except Exception as e:
            tqdm.write(f"  Error processing {wav}: {e}")

    # ===== Write index file =====
    list_path = os.path.join(output_dir, "whisper_encoder_cali.txt")
    with open(list_path, "w") as f:
        f.write("\n".join(calib_paths) + "\n")

    print(f"Encoder calibration: {len(calib_paths)} files -> {list_path}")
    return list_path


def build_decoder_sample(model, processor, audio_chunk, language, task):
    """Transcribe/translate one chunk and return its (encoder, decoder) hidden states as float32.

    Returns None if generation produced no tokens.
    """
    decoder = model.model.decoder
    features = processor(audio_chunk, sampling_rate=16000, return_tensors="pt").input_features.to(model.device)
    with torch.no_grad():
        encoder_hidden = model.model.encoder(features).last_hidden_state
        tokens = model.generate(features, language=language, task=task)[0]
        if tokens.numel() == 0:
            return None
        decoder_hidden = decoder.embed_tokens(tokens[None]) + decoder.embed_positions(tokens[None])
    return (
        encoder_hidden.cpu().numpy().astype(np.float32),
        decoder_hidden.cpu().numpy().astype(np.float32),
    )


def generate_decoder_calibration_data(model, processor, audio_dir, output_dir="./calibration_data/decoder", num_samples=1000):
    """Save decoder input embeddings (token + positional) as calibration data.

    Two independent knobs per clip:
      - length: cut the clip to several prefixes (FRACTIONS_TO_USE) for diverse sequence lengths.
      - task:   transcribe or translate each sample (20% translate, 80% transcribe).
    """
    print("Generating decoder calibration data...")
    os.makedirs(output_dir, exist_ok=True)

    # ===== Build samples: length prefix (FRACTIONS_TO_USE) x task per clip =====
    samples = []  # (decoder_path, encoder_path, task)
    idx = 0
    for wav in tqdm(list_wavs(audio_dir, num_samples), desc="Decoder calibration", unit="file"):
        language = detect_language(wav)
        if language is None:
            tqdm.write(f"  Skipping {wav}: language not in mapping")
            continue
        try:
            audio, _ = librosa.load(os.path.join(audio_dir, wav), sr=16000)
        except Exception as e:
            tqdm.write(f"  Error loading {wav}: {e}")
            continue

        for fraction in FRACTIONS_TO_USE:
            # Use the first `fraction` of the clip (shorter prefix -> shorter transcript).
            chunk = audio[: int(fraction * len(audio))]
            # 20% translation, 80% transcription (transcription is the primary task).
            task = "translate" if random.random() < 0.2 else "transcribe"

            try:
                hidden = build_decoder_sample(model, processor, chunk, language, task)
            except Exception as e:
                tqdm.write(f"  Error processing {wav} fraction={fraction}: {e}")
                continue
            if hidden is None:  # empty generation
                continue
            encoder_np, decoder_np = hidden
            encoder_path, decoder_path = save_sample(output_dir, idx, encoder_np, decoder_np)
            samples.append((decoder_path, encoder_path, task))
            idx += 1

    # ===== Write index file (info + calib paths) =====
    d_model = model.config.d_model
    calib_json = {
        "info": {
            "input names": ["decoder_hidden_states", "encoder_hidden_states"],
            "input shapes": [[1, -1, d_model], [1, 1500, d_model]],  # decoder is dynamic, encoder fixed
        },
        "calib paths": [[decoder_path, encoder_path] for decoder_path, encoder_path, _ in samples],
    }
    calib_json_path = os.path.join(output_dir, "whisper_decoder_calib.json")
    with open(calib_json_path, "w") as f:
        json.dump(calib_json, f, indent=2)

    print(f"Decoder calibration: {len(samples)} samples -> {calib_json_path}")
    task_counts = {}
    for _, _, task in samples:
        task_counts[task] = task_counts.get(task, 0) + 1
    print(f"Task distribution: {task_counts}")
    return calib_json_path


def main():
    audio_dir = "./audio_files"
    if not os.path.exists(audio_dir):
        print(f"Audio directory not found: {audio_dir}")
        print("Please run prepare_audio.py first!")
        return

    set_seed(42)
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model = model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

    encoder_calib_path = generate_encoder_calibration_data(processor, audio_dir)
    decoder_calib_path = generate_decoder_calibration_data(model, processor, audio_dir)

    print(f"\nEncoder calibration: {encoder_calib_path}")
    print(f"Decoder calibration: {decoder_calib_path}")


if __name__ == "__main__":
    main()
