#!/usr/bin/env python3
"""
Create Calibration Data using HuggingFace Transformers (V3)
Generates calibration datasets for both encoder and decoder using new datasets
Supports both transcription and translation cases with proper special tokens
Generates transcriptions and translations using whisper-small on-the-fly
"""

import os
import json
import numpy as np
import torch
import random
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# FLEURS to Whisper language mapping
FLEURS_TO_WHISPER = {
    'ar_eg': 'ar',
    'cmn_hans_cn': 'zh',
    'de_de': 'de',
    'el_gr': 'el',
    'en_us': 'en',
    'es_419': 'es',
    'fr_fr': 'fr',
    'id_id': 'id',
    'it_it': 'it',
    'ja_jp': 'ja',
    'ko_kr': 'ko',
    'ms_my': 'ms',
    'nl_nl': 'nl',
    'pt_br': 'pt',
    'ru_ru': 'ru',
    'th_th': 'th',
    'ur_pk': 'ur',
    'vi_vn': 'vi',
}

def create_encoder_calibration_data(audio_dir, output_dir="./encoder", num_samples=1000):
    """Create calibration data for Whisper encoder using HuggingFace preprocessing"""
    
    print("=" * 60)
    print("🎵 Creating Encoder Calibration Data (HuggingFace V3)")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load HuggingFace processor
    print("Loading HuggingFace Whisper processor...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    
    # Get audio files
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')][:num_samples]
    print(f"Processing {len(audio_files)} audio files...")
    
    calibration_files = []
    
    for i, audio_file in enumerate(audio_files):
        audio_path = os.path.join(audio_dir, audio_file)
        
        try:
            # Load audio file
            audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
            
            # Process with HuggingFace processor
            inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
            
            # Get the mel spectrogram (input_features)
            mel_features = inputs.input_features  # Shape: [1, 80, 3000]
            
            # Convert to numpy and ensure proper format
            mel_np = mel_features.transpose(1,2).cpu().numpy().astype(np.float32)
            mel_np = np.ascontiguousarray(mel_np)
            
            # Save calibration file
            calib_filename = f"encoder_calib_{i:04d}.npy"
            calib_path = os.path.join(output_dir, calib_filename)
            np.save(calib_path, mel_np)
            calibration_files.append(calib_path)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(audio_files)} files")
                
        except Exception as e:
            print(f"❌ Error processing {audio_file}: {e}")
            continue
    
    # Create calibration list file
    calib_list_path = os.path.join(output_dir, "whisper_encoder_cali.txt")
    with open(calib_list_path, 'w') as f:
        for calib_file in calibration_files:
            f.write(f"{os.path.abspath(calib_file)}\n")
    
    print(f"✅ Encoder calibration data created:")
    print(f"  Files: {len(calibration_files)}")
    print(f"  Directory: {output_dir}")
    print(f"  List file: {calib_list_path}")
    
    return calib_list_path


def create_decoder_calibration_data(audio_dir, output_dir="./decoder", num_samples=1000):
    """Create calibration data for Whisper decoder using HuggingFace preprocessing
    Generates transcriptions and translations on-the-fly using whisper-small
    """
    
    print("=" * 60)
    print("🎯 Creating Decoder Calibration Data (HuggingFace V3)")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load HuggingFace model and processor
    print("Loading HuggingFace Whisper model and processor...")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model = model.eval().cuda()
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    
    # Get audio files
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')][:num_samples]
    print(f"Processing {len(audio_files)} audio files...")
    
    calibration_data = []
    
    for i, audio_file in enumerate(audio_files):
        # if 'en_us' in audio_file:
        #     continue
        audio_path = os.path.join(audio_dir, audio_file)
        
        try:
            # Determine language from filename
            lang_code = None
            for fleurs_lang in FLEURS_TO_WHISPER.keys():
                if audio_file.startswith(fleurs_lang):
                    lang_code = FLEURS_TO_WHISPER[fleurs_lang]
                    break
            
            # Skip if language not found in mapping
            if lang_code is None:
                print(f"  ⚠️ Skipping {audio_file}: language not in FLEURS_TO_WHISPER mapping")
                continue
            
            # Process audio through encoder
            audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
            inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
            
            # Get encoder output
            with torch.no_grad():
                encoder_output = model.model.encoder(inputs.input_features.cuda())
                encoder_hidden_states = encoder_output.last_hidden_state.cpu().numpy().astype(np.float32)
            
            # Generate transcription and translation using whisper-small model
            print(f"  Generating transcription and translation for {audio_file}...")
            
            # Generate transcription
            with torch.no_grad():
                transcription_ids = model.generate(
                    inputs.input_features.cuda(),
                    language=lang_code,
                    task="transcribe"
                )
            transcription = processor.batch_decode(transcription_ids, skip_special_tokens=True)[0]
            
            # Generate translation
            with torch.no_grad():
                translation_ids = model.generate(
                    inputs.input_features.cuda(),
                    language=lang_code,
                    task="translate"
                )
            translation = processor.batch_decode(translation_ids, skip_special_tokens=True)[0]
            
            if not transcription:
                print(f"  ⚠️ No transcription generated for {audio_file}, skipping...")
                continue
            
            # Randomly choose between transcription and translation (50/50)
            use_translation = True if random.random() < 0.2 else False
            target_text = translation if use_translation and translation else transcription
            
            # Apply Whisper-style text normalization using HuggingFace processor
            # This helps ensure text fits Whisper's expected input style
            if target_text:
                # Use HuggingFace processor's built-in normalization
                # The processor handles proper text normalization for Whisper
                try:
                    # Normalize text using the processor's tokenizer normalization
                    normalized_text = processor.tokenizer.normalize(target_text)
                    target_text = normalized_text if normalized_text else target_text
                except:
                    # Fallback to basic normalization if processor method fails
                    target_text = target_text.lower().strip()
                    import re
                    target_text = re.sub(r'\s+', ' ', target_text)  # Normalize whitespace
            
            # Randomly omit language token 20% of the time
            #omit_language = random.random() < 0.2
            
            # Determine task type for metadata
            if use_translation:
                task_type = "translation_no_lang" if False else "translation_with_lang"
            else:
                task_type = "transcription_no_lang" if False else "transcription_with_lang"
            
            # Build proper token sequence with special tokens
            # Use processor's built-in method to properly handle special tokens
            tokenizer = processor.tokenizer
            
            # Set language and task for the tokenizer
            #if not omit_language:
            if True:
                tokenizer.language = lang_code
            else:
                tokenizer.language = None
            
            if use_translation:
                tokenizer.task = "translate"
            else:
                tokenizer.task = "transcribe"
            
            # Use processor to tokenize with proper special tokens
            # This will automatically add: SOT, language, task, timestamps tokens
            encoded = processor.tokenizer(
                target_text,
                add_special_tokens=True,
                return_tensors="pt"
            )
            tokens = encoded.input_ids[0].tolist()
            
            print(f"  Tokens: {tokens[:10]}... (total {len(tokens)})")
            print(f"  Decoded: {processor.tokenizer.decode(tokens)}")
            
            if len(tokens) > 4096:  # Truncate if too long
                tokens = tokens[:4096]
            
            # Create decoder hidden states (token embeddings + positional embeddings)
            tokens_tensor = torch.tensor([tokens], dtype=torch.long)
            token_embeds = model.model.decoder.embed_tokens(tokens_tensor.cuda())
            offset = 0  # No KV-cache during calibration
            positions = model.model.decoder.embed_positions(tokens_tensor.cuda())
            decoder_hidden_states = (token_embeds + positions).detach().cpu().numpy().astype(np.float32)
            
            # Save calibration files
            encoder_calib_path = os.path.join(os.path.abspath(output_dir), f"sample_{i:04d}", "encoder_hidden_states.npy")
            decoder_calib_path = os.path.join(os.path.abspath(output_dir), f"sample_{i:04d}", "decoder_hidden_states.npy")
            
            os.makedirs(os.path.dirname(encoder_calib_path), exist_ok=True)
            np.save(encoder_calib_path, encoder_hidden_states)
            np.save(decoder_calib_path, decoder_hidden_states)
            
            calibration_data.append({
                "encoder_hidden_states": encoder_calib_path,
                "decoder_hidden_states": decoder_calib_path,
                "task_type": task_type,
                "source_language": lang_code,
                "use_translation": use_translation,
                #"omit_language": omit_language,
                "decoded_tokens": processor.tokenizer.decode(tokens)
            })
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(audio_files)} files")
                
        except Exception as e:
            print(f"❌ Error processing {audio_file}: {e}")
            continue
    
    # Create calibration JSON file
    calib_json_path = os.path.join(output_dir, "whisper_decoder_calib.json")
    
    calib_json = {
        "info": {
            "input names": ["decoder_hidden_states", "encoder_hidden_states"],
            "input shapes": [
                [1, -1, 768],  # Dynamic shape for decoder
                [1, 1500, 768]  # Fixed shape for encoder
            ],
            "task_types": ["transcription_with_lang", "transcription_no_lang", "translation_with_lang", "translation_no_lang"],
            "language_omission_rate": 0.2
        },
        "calib paths": [[item["decoder_hidden_states"], item["encoder_hidden_states"]] for item in calibration_data],
        "metadata": calibration_data
    }
    
    with open(calib_json_path, 'w') as f:
        json.dump(calib_json, f, indent=2)
    with open(calib_json_path.replace('.json', '_metadata.json'), 'w') as f:
        json.dump([item["decoded_tokens"] for item in calibration_data], f, indent=2)

    print(f"✅ Decoder calibration data created:")
    print(f"  Samples: {len(calibration_data)}")
    print(f"  Directory: {output_dir}")
    print(f"  JSON file: {calib_json_path}")
    
    # Print statistics
    task_counts = {}
    for item in calibration_data:
        task_type = item["task_type"]
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    
    print(f"📊 Task distribution:")
    for task_type, count in task_counts.items():
        print(f"  {task_type}: {count}")
    
    return calib_json_path


def main():
    """Create calibration data for both encoder and decoder using new datasets"""
    
    print("🎯 Creating Calibration Data for Whisper Encoder and Decoder (HuggingFace V3)")
    print("=" * 70)
    
    # Check if data exists
    audio_dir = "../data/audio_files"
    
    if not os.path.exists(audio_dir):
        print(f"❌ Audio directory not found: {audio_dir}")
        print("Please run download_data.py first!")
        return 1
    
    # Create encoder calibration
    encoder_calib_path = create_encoder_calibration_data(audio_dir)
    
    # Create decoder calibration
    decoder_calib_path = create_decoder_calibration_data(audio_dir)
    
    print("\n" + "=" * 70)
    print("🎉 Calibration Data Creation Complete!")
    print("=" * 70)
    print(f"Encoder calibration: {encoder_calib_path}")
    print(f"Decoder calibration: {decoder_calib_path}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())

