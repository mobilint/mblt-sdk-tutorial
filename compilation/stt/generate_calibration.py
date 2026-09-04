import json
import random
from argparse import ArgumentParser
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_ID = "openai/whisper-small"
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
MIN_TOKENS = 8
TRANSLATE_RATIO = 0.2
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_audio_files(audio_dir: Path) -> list[Path]:
    nested_paths = sorted(path for path in audio_dir.rglob("*.wav") if path.parent != audio_dir)
    paths = nested_paths or sorted(audio_dir.glob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"No WAV files found in {audio_dir}")
    return paths


def detect_language(filename: str) -> str:
    for prefix, language in FLEURS_TO_WHISPER.items():
        if filename.startswith(prefix):
            return language
    raise ValueError(f"Unsupported FLEURS filename: {filename}")


def transcribe_clip(model, processor, audio: np.ndarray, language: str, task: str):
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(model.device)
    with torch.inference_mode():
        encoder_hidden_states = model.model.encoder(input_features).last_hidden_state
        generated_tokens = model.generate(input_features, language=language, task=task)[0]

    if generated_tokens.numel() == 0:
        return None

    tokenizer = processor.tokenizer
    tokenizer.set_prefix_tokens(language=language, task=task, predict_timestamps=False)
    prefix_tokens = torch.tensor(
        tokenizer.prefix_tokens,
        device=generated_tokens.device,
        dtype=generated_tokens.dtype,
    )
    tokens = torch.cat((prefix_tokens, generated_tokens))

    eos_token_id = model.generation_config.eos_token_id
    eos_token_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])
    if tokens[-1].item() in eos_token_ids:
        tokens = tokens[:-1]

    return encoder_hidden_states.cpu().numpy().astype(np.float32), tokens


def decoder_embeddings(model, tokens: torch.Tensor) -> np.ndarray:
    decoder = model.model.decoder
    token_ids = tokens.unsqueeze(0)
    with torch.inference_mode():
        embeddings = decoder.embed_tokens(token_ids) + decoder.embed_positions(token_ids)
    return embeddings.cpu().numpy().astype(np.float32)


def generate_encoder_calibration(processor, audio_paths: list[Path], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_paths = []

    for index, audio_path in enumerate(tqdm(audio_paths, desc="Encoder calibration", unit="file")):
        audio, _ = librosa.load(audio_path, sr=16000)
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        input_features = np.ascontiguousarray(input_features.transpose(1, 2).cpu().numpy().astype(np.float32))
        calibration_path = (output_dir / f"encoder_calib_{index:04d}.npy").resolve()
        np.save(calibration_path, input_features)
        calibration_paths.append(calibration_path)

    manifest_path = output_dir / "whisper_encoder_cali.txt"
    manifest_path.write_text("\n".join(map(str, calibration_paths)) + "\n", encoding="utf-8")
    return manifest_path


def generate_decoder_calibration(
    model,
    processor,
    audio_paths: list[Path],
    output_dir: Path,
    fractions: list[float],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_paths = []
    skipped_short = 0

    for audio_path in tqdm(audio_paths, desc="Decoder calibration", unit="file"):
        audio, _ = librosa.load(audio_path, sr=16000)
        language = detect_language(audio_path.name)

        for fraction in fractions:
            audio_chunk = audio[: int(len(audio) * fraction)]
            task = "translate" if random.random() < TRANSLATE_RATIO else "transcribe"
            result = transcribe_clip(model, processor, audio_chunk, language, task)
            if result is None:
                continue

            encoder_hidden_states, tokens = result
            if len(tokens) < MIN_TOKENS:
                skipped_short += 1
                continue

            sample_dir = (output_dir / f"sample_{len(calibration_paths):04d}").resolve()
            sample_dir.mkdir(parents=True, exist_ok=True)
            encoder_path = sample_dir / "encoder_hidden_states.npy"
            decoder_path = sample_dir / "decoder_hidden_states.npy"
            np.save(encoder_path, encoder_hidden_states)
            np.save(decoder_path, decoder_embeddings(model, tokens))
            calibration_paths.append([str(decoder_path), str(encoder_path)])

    if not calibration_paths:
        raise RuntimeError("No decoder calibration samples were generated")

    d_model = model.config.d_model
    manifest = {
        "info": {
            "input names": ["decoder_hidden_states", "encoder_hidden_states"],
            "input shapes": [[1, -1, d_model], [1, 1500, d_model]],
        },
        "calib paths": calibration_paths,
    }
    manifest_path = output_dir / "whisper_decoder_calib.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(calibration_paths)} decoder samples; skipped {skipped_short} short samples")
    return manifest_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=Path("./audio_files"))
    parser.add_argument("--output-dir", type=Path, default=Path("./calibration_data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS))
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.output_dir}")

    set_seed(args.seed)
    audio_files = find_audio_files(args.audio_dir)
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    encoder_manifest = generate_encoder_calibration(processor, audio_files, args.output_dir / "encoder")
    decoder_manifest = generate_decoder_calibration(
        model,
        processor,
        audio_files,
        args.output_dir / "decoder",
        args.fractions,
    )
    print(f"Encoder calibration: {encoder_manifest}")
    print(f"Decoder calibration: {decoder_manifest}")
