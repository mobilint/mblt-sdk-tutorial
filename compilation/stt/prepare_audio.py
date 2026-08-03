"""Download FLEURS audio data for calibration."""

import os

# Must run before importing datasets/huggingface_hub: these timeout constants are read
# from the env at import time. The 10s default times out slow reads and spams backoff
# retries; raise it.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from itertools import islice  # noqa: E402

import librosa  # noqa: E402
import soundfile as sf  # noqa: E402
from datasets import load_dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

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
    """Stream one FLEURS language into ``audio_dir`` as ``{lang}_{i:04d}.wav`` (16 kHz).

    Idempotent: skips existing files, and skips the network if already complete.
    Returns the number of newly written files.
    """
    # Skip the network if this language is already complete.
    if all(
        os.path.isfile(os.path.join(audio_dir, f"{lang}_{i:04d}.wav"))
        for i in range(num_samples_per_lang)
    ):
        return 0

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
    n_workers: int = 4,
) -> str:
    """Download FLEURS audio into ``audio_files/`` (flat layout), one thread per language.

    A modest ``n_workers`` is intentional: too many concurrent streams saturate the link
    and cause read timeouts. Existing files are skipped on re-run.
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

    # Skip finalization to avoid a PyGILState_Release crash from lingering streaming
    # threads. See https://github.com/huggingface/datasets/issues/7357
    os._exit(0)
