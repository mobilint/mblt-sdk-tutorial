#!/usr/bin/env python3
"""
Step 1: Download FLEURS Audio Data
Downloads audio files and transcriptions from Google/FLEURS dataset
"""

import os
import json
import numpy as np
from datasets import load_dataset
import soundfile as sf
import whisper
import librosa
import torch
import random


# FLEURS language codes for downloading data
FLEURS_LANGUAGES = [
    'ar_eg',   # Arabic (Egypt)
    'cmn_hans_cn',  # Mandarin Chinese (Simplified)
    'de_de',   # German
    'el_gr',   # Greek
    'en_us',   # English
    'es_419',  # Spanish (Latin America)
    'fr_fr',   # French
    'id_id',   # Indonesian
    'it_it',   # Italian
    'ja_jp',   # Japanese
    'ko_kr',   # Korean
    'pt_br',   # Portuguese (Brazil)
    'ru_ru',   # Russian
    'ta_in',   # Tamil
    'th_th',   # Thai
    'ur_pk',   # Urdu
    'vi_vn',   # Vietnamese
]

# Whisper language codes for translation
WHISPER_LANGUAGES = [
    'ar',   # Arabic
    'zh',   # Chinese
    'de',   # German
    'el',   # Greek
    'en',   # English
    'es',   # Spanish
    'fr',   # French
    'id',   # Indonesian
    'it',   # Italian
    'ja',   # Japanese
    'ko',   # Korean
    'ms',   # Malay
    'nl',   # Dutch
    'pt',   # Portuguese
    'ru',   # Russian
    'ta',   # Tamil
    'th',   # Thai
    'ur',   # Urdu
    'vi',   # Vietnamese
]

# Mapping from FLEURS codes to Whisper codes
FLEURS_TO_WHISPER = {
    'ar_eg': 'ar',
    'bg_bg': 'bg',
    'ca_es': 'ca',
    'cmn_hans_cn': 'zh',
    'yue_hant_hk': 'zh',  # Cantonese maps to Chinese in Whisper
    'cs_cz': 'cs',
    'da_dk': 'da',
    'de_de': 'de',
    'el_gr': 'el',
    'en_us': 'en',
    'es_419': 'es',
    'fi_fi': 'fi',
    'fr_fr': 'fr',
    'he_il': 'he',
    'hi_in': 'hi',
    'hu_hu': 'hu',
    'id_id': 'id',
    'it_it': 'it',
    'ja_jp': 'ja',
    'ko_kr': 'ko',
    'ms_my': 'ms',
    'nl_nl': 'nl',
    'pl_pl': 'pl',
    'pt_br': 'pt',
    'ro_ro': 'ro',
    'ru_ru': 'ru',
    'sv_se': 'sv',
    'ta_in': 'ta',
    'th_th': 'th',
    'tr_tr': 'tr',
    'uk_ua': 'uk',
    'ur_pk': 'ur',
    'vi_vn': 'vi',
}

def download_fleurs_data(output_dir=".", languages=FLEURS_LANGUAGES, num_samples_per_lang=20):
    """Download FLEURS audio data, transcriptions, and translations"""
    
    print("=" * 60)
    print("📥 Downloading FLEURS Audio Data")
    print("=" * 60)
    print(f"Languages: {languages}")
    print(f"Samples per language: {num_samples_per_lang}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Create output directories
    audio_dir = os.path.join(output_dir, "audio_files")
    os.makedirs(audio_dir, exist_ok=True)
    
    # Load Whisper Large V3 for transcriptions and translations
    print("🤖 Loading Whisper Large V3 for transcriptions and translations...")
    model = whisper.load_model("large-v3").eval()
    if torch.cuda.is_available():
        model = model.cuda()
    print("✅ Whisper model loaded")
    
    transcriptions = {}
    translations = {}
    
    # Set random seed for reproducible random target language selection
    random.seed(42)
    
    for lang in languages:
        print(f"\n🌍 Downloading {lang} data...")
        
        try:
            # Load dataset
            dataset = load_dataset("google/fleurs", lang, split="validation", trust_remote_code=True)
            print(f"✅ Loaded {len(dataset)} samples for {lang}")
            
            # Download first num_samples_per_lang samples
            for i in range(min(num_samples_per_lang, len(dataset))):
                sample = dataset[i]
                
                # Save audio file
                audio_filename = f"{lang}_{i:04d}.wav"
                audio_path = os.path.join(audio_dir, audio_filename)
                
                # Convert to 16kHz mono if needed
                audio = sample["audio"]["array"]
                sample_rate = sample["audio"]["sampling_rate"]
                
                if sample_rate != 16000:
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                
                # Save as WAV file
                sf.write(audio_path, audio, 16000)
                
                # Convert FLEURS language code to Whisper language code
                source_whisper_lang = FLEURS_TO_WHISPER.get(lang, 'en')
                
                # Generate transcription using OpenAI Whisper
                try:
                    print(f"  Transcribing {audio_filename} in {lang} ({source_whisper_lang})...")
                    result = model.transcribe(audio_path, task="transcribe", language=source_whisper_lang)
                    transcriptions[audio_filename] = result["text"]
                except Exception as e:
                    print(f"  ⚠️ Transcription failed for {audio_filename}: {e}")
                    # Fallback to original transcription if Whisper fails
                    transcriptions[audio_filename] = sample["transcription"]
                
                # Generate English translation using OpenAI Whisper
                try:
                    print(f"  Translating {audio_filename} from {lang} ({source_whisper_lang}) to English...")
                    
                    # Whisper only supports translation to English
                    # Use translate task with source language specified
                    result = model.transcribe(audio_path, task="translate", language=source_whisper_lang)
                    
                    translations[audio_filename] = {
                        "source_language_fleurs": lang,
                        "source_language_whisper": source_whisper_lang,
                        "target_language": "en",
                        "translation": result["text"]
                    }
                except Exception as e:
                    print(f"  ⚠️ Translation failed for {audio_filename}: {e}")
                    translations[audio_filename] = {
                        "source_language_fleurs": lang,
                        "source_language_whisper": source_whisper_lang,
                        "target_language": "en",
                        "translation": ""
                    }
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{num_samples_per_lang} samples")
            
            print(f"✅ Downloaded {min(num_samples_per_lang, len(dataset))} samples for {lang}")
            
        except Exception as e:
            print(f"❌ Error downloading {lang}: {e}")
            continue
    
    # Save transcriptions
    transcriptions_path = os.path.join(output_dir, "transcriptions.json")
    with open(transcriptions_path, 'w', encoding='utf-8') as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=2)
    
    # Save translations
    translations_path = os.path.join(output_dir, "translations.json")
    with open(translations_path, 'w', encoding='utf-8') as f:
        json.dump(translations, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved transcriptions to: {transcriptions_path}")
    print(f"✅ Saved translations to: {translations_path}")
    print(f"📊 Total audio files: {len(transcriptions)}")
    print(f"📁 Audio directory: {audio_dir}")
    
    return audio_dir, transcriptions_path, translations_path


if __name__ == "__main__":
    audio_dir, transcriptions_path, translations_path = download_fleurs_data()
    print(f"\n🎉 Data download complete!")
    print(f"Audio files: {audio_dir}")
    print(f"Transcriptions: {transcriptions_path}")
    print(f"Translations: {translations_path}")
