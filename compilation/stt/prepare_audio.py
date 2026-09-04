import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path

import librosa
import soundfile as sf
from datasets import load_dataset
from huggingface_hub import constants as hf_constants
from tqdm import tqdm

hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = 60
hf_constants.HF_HUB_ETAG_TIMEOUT = 30

FLEURS_LANGUAGES = {
    "ar_eg": "Arabic",
    "cmn_hans_cn": "Chinese",
    "de_de": "German",
    "el_gr": "Greek",
    "en_us": "English",
    "es_419": "Spanish",
    "fr_fr": "French",
    "id_id": "Indonesian",
    "it_it": "Italian",
    "ja_jp": "Japanese",
    "ko_kr": "Korean",
    "pt_br": "Portuguese",
    "ru_ru": "Russian",
    "ta_in": "Tamil",
    "th_th": "Thai",
    "ur_pk": "Urdu",
    "vi_vn": "Vietnamese",
}


def download_language(language_code: str, language_name: str, audio_root: Path, sample_count: int) -> int:
    language_dir = audio_root / language_name
    language_dir.mkdir(parents=True, exist_ok=True)

    expected_files = [language_dir / f"{language_code}_{index:04d}.wav" for index in range(sample_count)]
    expected_files += [path.with_suffix(".txt") for path in expected_files]
    if all(path.is_file() for path in expected_files):
        return 0

    dataset = load_dataset(
        "google/fleurs",
        language_code,
        split="validation",
        trust_remote_code=True,
        streaming=True,
    )

    written = 0
    for index, sample in enumerate(islice(dataset, sample_count)):
        wav_path = language_dir / f"{language_code}_{index:04d}.wav"
        text_path = wav_path.with_suffix(".txt")

        if not wav_path.is_file():
            audio = sample["audio"]["array"]
            sample_rate = sample["audio"]["sampling_rate"]
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sf.write(wav_path, audio, 16000, subtype="PCM_16")
            written += 1

        if not text_path.is_file():
            text_path.write_text(str(sample["transcription"]).strip(), encoding="utf-8")
            written += 1

    missing = [path for path in expected_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"Failed to prepare {language_code}: {len(missing)} files are missing")
    return written


def prepare_audio(output_dir: Path, samples_per_language: int, workers: int) -> Path:
    audio_root = output_dir / "audio_files"
    audio_root.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_language,
                language_code,
                language_name,
                audio_root,
                samples_per_language,
            ): language_code
            for language_code, language_name in FLEURS_LANGUAGES.items()
        }
        with tqdm(total=len(futures), desc="Languages", unit="language") as progress:
            for future in as_completed(futures):
                language_code = futures[future]
                progress.set_postfix(language=language_code)
                future.result()
                progress.update(1)

    return audio_root


if __name__ == "__main__":
    audio_dir = prepare_audio(Path("."), samples_per_language=20, workers=4)
    print(f"Prepared audio data in {audio_dir}")
    sys.stdout.flush()
    os._exit(0)
