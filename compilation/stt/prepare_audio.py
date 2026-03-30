"""Download FLEURS audio data for calibration."""

import os

import librosa
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

FLEURS_LANGUAGES = [
    "ar_eg",  # Arabic (Egypt)
    "cmn_hans_cn",  # Mandarin Chinese (Simplified)
    "de_de",  # German
    "el_gr",  # Greek
    "en_us",  # English
    "es_419",  # Spanish (Latin America)
    "fr_fr",  # French
    "id_id",  # Indonesian
    "it_it",  # Italian
    "ja_jp",  # Japanese
    "ko_kr",  # Korean
    "pt_br",  # Portuguese (Brazil)
    "ru_ru",  # Russian
    "ta_in",  # Tamil
    "th_th",  # Thai
    "ur_pk",  # Urdu
    "vi_vn",  # Vietnamese
]


def download_fleurs_data(output_dir=".", languages=FLEURS_LANGUAGES, num_samples_per_lang=20):
    """Download FLEURS audio data as 16kHz WAV files."""

    print(f"Downloading FLEURS data: {len(languages)} languages, {num_samples_per_lang} samples each")

    audio_dir = os.path.join(output_dir, "audio_files")
    os.makedirs(audio_dir, exist_ok=True)

    total_downloaded = 0
    lang_pbar = tqdm(languages, desc="Languages", unit="lang")

    for lang in lang_pbar:
        lang_pbar.set_postfix(lang=lang)

        try:
            dataset = load_dataset(
                "google/fleurs",
                lang,
                split="validation",
                trust_remote_code=True,
                streaming=True,
            )

            i = 0
            for sample in dataset:
                if i >= num_samples_per_lang:
                    break

                audio_filename = f"{lang}_{i:04d}.wav"
                audio_path = os.path.join(audio_dir, audio_filename)

                audio = sample["audio"]["array"]
                sample_rate = sample["audio"]["sampling_rate"]

                if sample_rate != 16000:
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

                sf.write(audio_path, audio, 16000)
                total_downloaded += 1
                i += 1

        except Exception as e:
            tqdm.write(f"Error downloading {lang}: {e}")
            continue

    print(f"\nTotal audio files: {total_downloaded}")
    print(f"Audio directory: {audio_dir}")

    return audio_dir


if __name__ == "__main__":
    audio_dir = download_fleurs_data()
    print("\nData download complete!")
    print(f"Audio files: {audio_dir}")

    # Skip Python finalization to avoid PyGILState_Release crash
    # caused by lingering PyArrow/aiohttp threads from streaming datasets.
    # See: https://github.com/huggingface/datasets/issues/7357
    os._exit(0)
