"""
Dataset Augmentation and Packaging Orchestrator
Generates 4 variations (Clean, 15dB SNR, 10dB SNR, 5dB SNR) and packages dataset/final_train/
"""

import os
import io
import csv
import uuid
import random
import argparse
import datetime
import sqlite3
from typing import List, Dict

import numpy as np
import soundfile as sf
from .snr_mixer import overlay_noise

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DEFAULT_NOISE_DIR = os.path.join(BASE_DIR, "dataset", "noise_banks")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "final_train")
SQLITE_PATH = os.path.join(BASE_DIR, "metadata.db")

def init_final_catalog_db():
    conn = sqlite3.connect(SQLITE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS final_train_catalog (
            file_id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            speaker_name TEXT NOT NULL,
            word_spoken TEXT NOT NULL,
            category TEXT NOT NULL,
            augmentation_type TEXT NOT NULL,
            snr_db TEXT NOT NULL,
            duration_ms INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def discover_clean_audio_files(base_dataset_dir: str) -> List[Dict]:
    """Discovers all collected human recordings and synthetic samples."""
    samples = []
    
    # 1. Human Collected Samples
    for root, _, files in os.walk(base_dataset_dir):
        if "noise_banks" in root or "final_train" in root or "synthetic" in root:
            continue
        for file in files:
            if not file.endswith(".wav") or "_noisy" in file:
                continue
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, base_dataset_dir).replace("\\", "/")
            parts = rel_path.split("/")
            
            category = "general_negative"
            if "trigger_word" in parts:
                category = "trigger_word"
            elif "rhyming" in parts:
                category = "rhyming_word"

            speaker = parts[-2] if len(parts) >= 2 else "anonymous"
            fn_parts = file.replace(".wav", "").split("_")
            word = fn_parts[1] if len(fn_parts) >= 2 else "unknown"

            samples.append({
                "source_path": full_path,
                "filename": file,
                "speaker": speaker,
                "word": word,
                "category": category,
                "is_synthetic": False
            })

    # 2. Synthetic TTS Samples
    synth_dir = os.path.join(base_dataset_dir, "synthetic")
    if os.path.exists(synth_dir):
        for root, _, files in os.walk(synth_dir):
            for file in files:
                if not file.endswith(".wav"):
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, synth_dir).replace("\\", "/")
                parts = rel_path.split("/")

                category = "general_negative"
                if "trigger_word" in parts:
                    category = "trigger_word"
                elif "rhyming" in parts:
                    category = "rhyming_word"

                speaker = parts[-2] if len(parts) >= 2 else "synthetic"
                fn_parts = file.replace(".wav", "").split("_")
                word = fn_parts[2] if len(fn_parts) >= 3 else "unknown"

                samples.append({
                    "source_path": full_path,
                    "filename": file,
                    "speaker": speaker,
                    "word": word,
                    "category": category,
                    "is_synthetic": True
                })

    return samples

def run_augmentation_pipeline(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    noise_dir: str = DEFAULT_NOISE_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    dry_run: bool = False,
    sample_limit: int = 0
):
    print("=" * 70)
    print("Audio Noise Augmentation & Dataset Packaging Pipeline (noise_pipeline)")
    print(f"Clean Audio Source: {dataset_dir}")
    print(f"Noise Banks Dir:    {noise_dir}")
    print(f"Output Directory:   {output_dir}")
    print("=" * 70)

    init_final_catalog_db()

    # Discover Noise Files
    fan_noise = os.path.join(noise_dir, "fan_hum_motor.wav")
    chatter_noise = os.path.join(noise_dir, "ambient_room_chatter_typing.wav")
    clap_noise = os.path.join(noise_dir, "sudden_hand_claps.wav")
    knock_noise = os.path.join(noise_dir, "door_knocks_desk_taps.wav")

    all_noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith(".wav")] if os.path.exists(noise_dir) else []
    if not all_noise_files:
        raise FileNotFoundError(f"No noise .wav files found in {noise_dir}. Run download_noise_banks.py first.")

    fan_noise = fan_noise if os.path.exists(fan_noise) else all_noise_files[0]
    chatter_noise = chatter_noise if os.path.exists(chatter_noise) else all_noise_files[min(1, len(all_noise_files)-1)]
    clap_noise = clap_noise if os.path.exists(clap_noise) else all_noise_files[min(2, len(all_noise_files)-1)]

    clean_samples = discover_clean_audio_files(dataset_dir)
    if sample_limit > 0:
        clean_samples = clean_samples[:sample_limit]

    print(f"\n[DISCOVERY] Discovered {len(clean_samples)} clean source audio recordings.")
    
    counts = {}
    for s in clean_samples:
        counts[s["category"]] = counts.get(s["category"], 0) + 1
    print(f" Breakdown: {counts}")

    total_augmented_clips = len(clean_samples) * 4
    print(f"\n[PLAN] Generating 4 Variations Per Clean Sample:")
    print(f" 1. Clean Sample (Original)            => {len(clean_samples)} files")
    print(f" 2. Low Noise @ 15dB SNR (Fan Hum)     => {len(clean_samples)} files")
    print(f" 3. Med Noise @ 10dB SNR (Room Chatter)=> {len(clean_samples)} files")
    print(f" 4. High Noise @ 5dB SNR (Sudden Claps)=> {len(clean_samples)} files")
    print(f" Total Final Training Samples: {total_augmented_clips}")

    if dry_run:
        print("\n[DRY RUN] Plan verified successfully. No files written.")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "trigger_word"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "negative_word", "rhyming"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "negative_word", "general"), exist_ok=True)

    catalog_rows = []
    processed_count = 0
    start_time = datetime.datetime.now()

    for item in clean_samples:
        clean_path = item["source_path"]
        speaker = item["speaker"]
        word = item["word"]
        category = item["category"]

        if category == "trigger_word":
            sub_folder = "trigger_word"
        elif category == "rhyming_word":
            sub_folder = os.path.join("negative_word", "rhyming")
        else:
            sub_folder = os.path.join("negative_word", "general")

        dest_dir = os.path.join(output_dir, sub_folder, speaker)
        os.makedirs(dest_dir, exist_ok=True)

        base_name = os.path.splitext(item["filename"])[0]

        audio_data, sr = sf.read(clean_path)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        duration_ms = int(len(audio_data) / sr * 1000)
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 1: Clean Original
        v1_filename = f"{base_name}_clean.wav"
        v1_path = os.path.join(dest_dir, v1_filename)
        v1_rel = os.path.relpath(v1_path, BASE_DIR).replace("\\", "/")
        sf.write(v1_path, audio_data, sr, subtype="PCM_16", format="WAV")
        catalog_rows.append({
            "file_id": str(uuid.uuid4()),
            "file_path": v1_rel,
            "speaker_name": speaker,
            "word_spoken": word,
            "category": category,
            "augmentation_type": "clean_original",
            "snr_db": "none",
            "duration_ms": duration_ms,
            "created_at": now_iso
        })

        # 2: Low Noise (15 dB SNR)
        v2_filename = f"{base_name}_fan_15db.wav"
        v2_path = os.path.join(dest_dir, v2_filename)
        v2_rel = os.path.relpath(v2_path, BASE_DIR).replace("\\", "/")
        overlay_noise(clean_path, fan_noise, v2_path, snr_gain=15, sr=sr)
        catalog_rows.append({
            "file_id": str(uuid.uuid4()),
            "file_path": v2_rel,
            "speaker_name": speaker,
            "word_spoken": word,
            "category": category,
            "augmentation_type": "low_noise_fan",
            "snr_db": "15dB",
            "duration_ms": duration_ms,
            "created_at": now_iso
        })

        # 3: Medium Noise (10 dB SNR)
        v3_filename = f"{base_name}_chatter_10db.wav"
        v3_path = os.path.join(dest_dir, v3_filename)
        v3_rel = os.path.relpath(v3_path, BASE_DIR).replace("\\", "/")
        overlay_noise(clean_path, chatter_noise, v3_path, snr_gain=10, sr=sr)
        catalog_rows.append({
            "file_id": str(uuid.uuid4()),
            "file_path": v3_rel,
            "speaker_name": speaker,
            "word_spoken": word,
            "category": category,
            "augmentation_type": "medium_noise_chatter",
            "snr_db": "10dB",
            "duration_ms": duration_ms,
            "created_at": now_iso
        })

        # 4: High Noise (5 dB SNR)
        impulse_noise = clap_noise if random.random() > 0.5 else knock_noise
        v4_filename = f"{base_name}_impulse_5db.wav"
        v4_path = os.path.join(dest_dir, v4_filename)
        v4_rel = os.path.relpath(v4_path, BASE_DIR).replace("\\", "/")
        overlay_noise(clean_path, impulse_noise, v4_path, snr_gain=5, sr=sr)
        catalog_rows.append({
            "file_id": str(uuid.uuid4()),
            "file_path": v4_rel,
            "speaker_name": speaker,
            "word_spoken": word,
            "category": category,
            "augmentation_type": "high_noise_impulse",
            "snr_db": "5dB",
            "duration_ms": duration_ms,
            "created_at": now_iso
        })

        processed_count += 1
        if processed_count % 20 == 0 or processed_count == len(clean_samples):
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            rate = (processed_count * 4) / max(1.0, elapsed)
            print(f"Augmented: {processed_count}/{len(clean_samples)} ({processed_count*4}/{total_augmented_clips} variations) | {rate:.1f} clips/sec")

    # Save CSV & SQLite
    train_catalog_csv = os.path.join(output_dir, "train_catalog.csv")
    with open(train_catalog_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_id", "file_path", "speaker_name", "word_spoken",
            "category", "augmentation_type", "snr_db", "duration_ms", "created_at"
        ])
        for row in catalog_rows:
            writer.writerow([
                row["file_id"], row["file_path"], row["speaker_name"], row["word_spoken"],
                row["category"], row["augmentation_type"], row["snr_db"], row["duration_ms"], row["created_at"]
            ])

    conn = sqlite3.connect(SQLITE_PATH)
    cursor = conn.cursor()
    for row in catalog_rows:
        cursor.execute("""
            INSERT OR REPLACE INTO final_train_catalog
            (file_id, file_path, speaker_name, word_spoken, category, augmentation_type, snr_db, duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["file_id"], row["file_path"], row["speaker_name"], row["word_spoken"],
            row["category"], row["augmentation_type"], row["snr_db"], row["duration_ms"], row["created_at"]
        ))
    conn.commit()
    conn.close()

    print(f"\n[SUCCESS] Final Dataset Preparation Complete!")
    print(f" Directory:    {output_dir}")
    print(f" Total Clips:  {len(catalog_rows)}")
    print(f" Catalog CSV:  {train_catalog_csv}")
    print(f" SQLite Table: final_train_catalog in {SQLITE_PATH}")

def main():
    parser = argparse.ArgumentParser(description="Acoustic Noise Augmentation Pipeline")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="Source dataset root directory")
    parser.add_argument("--noise-dir", default=DEFAULT_NOISE_DIR, help="Noise banks directory")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Final training dataset output directory")
    parser.add_argument("--sample-limit", type=int, default=0, help="Optional limit of source samples to augment")
    parser.add_argument("--dry-run", action="store_true", help="Preview augmentation plan without generating files")
    args = parser.parse_args()

    run_augmentation_pipeline(
        dataset_dir=args.dataset_dir,
        noise_dir=args.noise_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        sample_limit=args.sample_limit
    )

if __name__ == "__main__":
    main()
