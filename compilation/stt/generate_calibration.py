"""Generate calibration data for Whisper encoder and decoder."""

import json
import os
import random

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

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


def generate_encoder_calibration_data(audio_dir, output_dir="./calibration_data/encoder", num_samples=1000):
    """Generate calibration data for Whisper encoder."""

    print("Generating encoder calibration data...")
    os.makedirs(output_dir, exist_ok=True)

    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")][:num_samples]
    print(f"Processing {len(audio_files)} audio files...")

    calibration_files = []

    for i, audio_file in enumerate(tqdm(audio_files, desc="Encoder calibration", unit="file")):
        audio_path = os.path.join(audio_dir, audio_file)

        try:
            audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
            inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")

            # mel spectrogram: [1, 80, 3000] -> [1, 3000, 80]
            mel_np = inputs.input_features.transpose(1, 2).cpu().numpy().astype(np.float32)
            mel_np = np.ascontiguousarray(mel_np)

            calib_filename = f"encoder_calib_{i:04d}.npy"
            calib_path = os.path.join(output_dir, calib_filename)
            np.save(calib_path, mel_np)
            calibration_files.append(calib_path)

        except Exception as e:
            tqdm.write(f"  Error processing {audio_file}: {e}")
            continue

    calib_list_path = os.path.join(output_dir, "whisper_encoder_cali.txt")
    with open(calib_list_path, "w") as f:
        for calib_file in calibration_files:
            f.write(f"{os.path.abspath(calib_file)}\n")

    print(f"Encoder calibration: {len(calibration_files)} files -> {calib_list_path}")
    return calib_list_path


def generate_decoder_calibration_data(audio_dir, output_dir="./calibration_data/decoder", num_samples=1000):
    """Generate calibration data for Whisper decoder.

    Uses whisper-small to generate transcriptions and translations on-the-fly,
    then creates decoder input embeddings (token + positional) as calibration data.
    """

    print("Generating decoder calibration data...")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model = model.eval().to(device)
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")][:num_samples]
    print(f"Processing {len(audio_files)} audio files...")

    calibration_data = []

    for i, audio_file in enumerate(tqdm(audio_files, desc="Decoder calibration", unit="file")):
        audio_path = os.path.join(audio_dir, audio_file)

        try:
            # Determine language from filename
            lang_code = None
            for fleurs_lang in FLEURS_TO_WHISPER:
                if audio_file.startswith(fleurs_lang):
                    lang_code = FLEURS_TO_WHISPER[fleurs_lang]
                    break

            if lang_code is None:
                tqdm.write(f"  Skipping {audio_file}: language not in mapping")
                continue

            audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
            inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")

            # Get encoder output
            with torch.no_grad():
                encoder_output = model.model.encoder(inputs.input_features.to(device))
                encoder_hidden_states = (
                    encoder_output.last_hidden_state.cpu().numpy().astype(np.float32)
                )

            # 20% translation, 80% transcription
            # Reflects typical usage where transcription is the primary task.
            # Adjust ratio based on deployment needs (e.g., 0.5 if translation is equally used).
            use_translation = random.random() < 0.2
            task = "translate" if use_translation else "transcribe"

            with torch.no_grad():
                generated_ids = model.generate(
                    inputs.input_features.to(device), language=lang_code, task=task
                )
            tokens = generated_ids[0].tolist()

            # Skip if only special tokens were generated (no actual content)
            if len(tokens) <= 4:
                tqdm.write(f"  Skipping {audio_file}: no content generated")
                continue

            task_type = "translation_with_lang" if use_translation else "transcription_with_lang"

            # Create decoder hidden states (token embeddings + positional embeddings)
            # Uses generated token IDs directly to avoid decode→normalize→re-tokenize mismatch
            tokens_tensor = torch.tensor([tokens], dtype=torch.long)
            token_embeds = model.model.decoder.embed_tokens(tokens_tensor.to(device))
            positions = model.model.decoder.embed_positions(tokens_tensor.to(device))
            decoder_hidden_states = (
                (token_embeds + positions).detach().cpu().numpy().astype(np.float32)
            )

            # Save calibration files
            encoder_calib_path = os.path.join(
                os.path.abspath(output_dir), f"sample_{i:04d}", "encoder_hidden_states.npy",
            )
            decoder_calib_path = os.path.join(
                os.path.abspath(output_dir), f"sample_{i:04d}", "decoder_hidden_states.npy",
            )

            os.makedirs(os.path.dirname(encoder_calib_path), exist_ok=True)
            np.save(encoder_calib_path, encoder_hidden_states)
            np.save(decoder_calib_path, decoder_hidden_states)

            calibration_data.append(
                {
                    "encoder_hidden_states": encoder_calib_path,
                    "decoder_hidden_states": decoder_calib_path,
                    "task_type": task_type,
                    "source_language": lang_code,
                    "use_translation": use_translation,
                    "decoded_tokens": processor.tokenizer.decode(tokens),
                }
            )

        except Exception as e:
            tqdm.write(f"  Error processing {audio_file}: {e}")
            continue

    # Save calibration JSON
    calib_json_path = os.path.join(output_dir, "whisper_decoder_calib.json")

    calib_json = {
        "info": {
            "input names": ["decoder_hidden_states", "encoder_hidden_states"],
            "input shapes": [
                [1, -1, 768],  # Dynamic shape for decoder
                [1, 1500, 768],  # Fixed shape for encoder
            ],
        },
        "calib paths": [
            [item["decoder_hidden_states"], item["encoder_hidden_states"]]
            for item in calibration_data
        ],
        "metadata": calibration_data,
    }

    with open(calib_json_path, "w") as f:
        json.dump(calib_json, f, indent=2)
    with open(calib_json_path.replace(".json", "_metadata.json"), "w") as f:
        json.dump([item["decoded_tokens"] for item in calibration_data], f, indent=2)

    print(f"Decoder calibration: {len(calibration_data)} samples -> {calib_json_path}")

    # Task distribution
    task_counts = {}
    for item in calibration_data:
        task_counts[item["task_type"]] = task_counts.get(item["task_type"], 0) + 1
    print(f"Task distribution: {task_counts}")

    return calib_json_path


if __name__ == "__main__":
    audio_dir = "./audio_files"

    if not os.path.exists(audio_dir):
        print(f"Audio directory not found: {audio_dir}")
        print("Please run prepare_audio.py first!")
    else:
        encoder_calib_path = generate_encoder_calibration_data(audio_dir)
        decoder_calib_path = generate_decoder_calibration_data(audio_dir)

        print(f"\nEncoder calibration: {encoder_calib_path}")
        print(f"Decoder calibration: {decoder_calib_path}")
