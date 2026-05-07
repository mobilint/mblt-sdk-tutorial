"""Download FLEURS audio data for calibration."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

import librosa
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

FLEURS_LANGUAGES: list[str] = [
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


def download_one_language(lang: str, audio_dir: str, num_samples_per_lang: int) -> int:
    """Stream one FLEURS language and save WAV files into ``audio_dir`` as a flat layout.

    Samples whose WAV file already exists are skipped (idempotent). Audio that is
    not 16 kHz is resampled with ``librosa.resample`` before being written.

    Args:
        lang: FLEURS language code (e.g. ``ko_kr``).
        audio_dir: Directory where ``{lang}_{i:04d}.wav`` files are saved.
        num_samples_per_lang: Number of samples to fetch.

    Returns:
        Count of WAV files newly written to disk (skipped files excluded).
    """
    new_count = 0
    try:
        dataset = load_dataset(
            "google/fleurs",
            lang,
            split="validation",
            trust_remote_code=True,
            streaming=True,
        )
        for i, sample in enumerate(islice(dataset, num_samples_per_lang)):
            wav_path = os.path.join(audio_dir, f"{lang}_{i:04d}.wav")
            if os.path.isfile(wav_path):
                continue

            audio = sample["audio"]["array"]
            sample_rate = sample["audio"]["sampling_rate"]
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

            sf.write(wav_path, audio, 16000)
            new_count += 1
    except Exception as e:
        tqdm.write(f"Error downloading {lang}: {e}")
    return new_count


def download_fleurs_data(
    output_dir: str = ".",
    languages: list[str] = FLEURS_LANGUAGES,
    num_samples_per_lang: int = 20,
    n_workers: int = 8,
) -> str:
    """Download FLEURS audio into ``audio_files/`` with a flat layout.

    Languages are fetched concurrently via ``ThreadPoolExecutor`` (max_workers=n_workers).
    Each per-language streaming iterator is consumed by a single thread and stays
    thread-safe. On re-run, samples whose WAV file already exists are skipped.

    Args:
        output_dir: Parent directory in which ``audio_files/`` is created.
        languages: FLEURS language codes to fetch.
        num_samples_per_lang: Samples per language.
        n_workers: Maximum number of languages downloaded concurrently.

    Returns:
        Path to the ``audio_files`` directory.
    """
    print(
        f"Downloading FLEURS data: {len(languages)} languages, "
        f"{num_samples_per_lang} samples each, {n_workers} parallel workers"
    )

    audio_dir = os.path.join(output_dir, "audio_files")
    os.makedirs(audio_dir, exist_ok=True)

    total_new = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(download_one_language, lang, audio_dir, num_samples_per_lang): lang for lang in languages
        }
        with tqdm(total=len(languages), desc="Languages", unit="lang") as pbar:
            for fut in as_completed(futures):
                lang = futures[fut]
                pbar.set_postfix(lang=lang)
                total_new += fut.result()
                pbar.update(1)

    print(f"\nNew files written: {total_new} (skipped existing)")
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
