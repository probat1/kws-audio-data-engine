"""
Synthetic Dataset Generation Orchestrator
Iterates through parameter variation matrix and creates standardized 1.0s WAV samples in dataset/synthetic/
"""

import os
import io
import re
import csv
import json
import uuid
import asyncio
import argparse
import datetime
import sqlite3
import random
from typing import Dict

from .voices_config import VOICE_PROFILES, SPEED_RATES, PITCH_ADJUSTMENTS, VOLUME_GAINS_DB
from .tts_engine import generate_single_tts_clip

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "words_config.json")
SYNTHETIC_DIR = os.path.join(BASE_DIR, "dataset", "synthetic")
SQLITE_PATH = os.path.join(BASE_DIR, "metadata.db")
CSV_PATH = os.path.join(BASE_DIR, "dataset_catalog.csv")

def sanitize_for_filename(text: str) -> str:
    """Sanitizes words/phrases for file naming."""
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(text).strip())
    clean = re.sub(r"_+", "_", clean).strip("_").lower()
    return clean or "word"

def load_words_config() -> Dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "trigger_word": "Hey Nexus",
        "rhyming_words": ["Hey", "Nexus", "Say Lexus", "Play Texas"],
        "negative_words": ["Turn on the lights", "Turn off the fan", "Set a timer"]
    }

async def run_synthetic_pipeline(
    trigger_target: int = 500,
    rhyme_target: int = 250,
    negative_target: int = 250,
    concurrency: int = 12,
    dry_run: bool = False
):
    print("=" * 70)
    print("TTS Synthetic Dataset Generation Pipeline (tts_pipeline)")
    print(f"Target Keyword Samples: {trigger_target}")
    print(f"Target Rhyming Samples: {rhyme_target}")
    print(f"Target Negative Samples: {negative_target}")
    print(f"Concurrency: {concurrency} workers")
    print("=" * 70)

    config = load_words_config()
    trigger_word = config.get("trigger_word", "Hey Nexus")
    rhyming_words = config.get("rhyming_words", ["Hey", "Nexus", "Say Lexus", "Play Texas"])
    negative_words = config.get("negative_words", ["Turn on the lights", "Turn off the fan"])

    tasks_to_run = []
    
    # 1. Build Keyword Tasks distributed evenly across all voices & parameters
    safe_trigger = sanitize_for_filename(trigger_word)
    for idx in range(trigger_target):
        voice_obj = VOICE_PROFILES[idx % len(VOICE_PROFILES)]
        voice_id = voice_obj["id"]
        rate = SPEED_RATES[(idx // len(VOICE_PROFILES)) % len(SPEED_RATES)]
        pitch = PITCH_ADJUSTMENTS[(idx // (len(VOICE_PROFILES) * len(SPEED_RATES))) % len(PITCH_ADJUSTMENTS)]
        gain = VOLUME_GAINS_DB[idx % len(VOLUME_GAINS_DB)]
        
        speaker_name = f"synthetic_{voice_id.split('-')[-1]}"
        filename = f"{speaker_name}_{safe_trigger}_silent_{idx+1:04d}.wav"
        folder = os.path.join("trigger_word", voice_id)
        rel_path = os.path.join("dataset", "synthetic", folder, filename).replace("\\", "/")
        full_path = os.path.join(BASE_DIR, rel_path)
        
        tasks_to_run.append({
            "word": trigger_word,
            "category": "trigger_word",
            "folder": folder,
            "voice": voice_id,
            "rate": rate,
            "pitch": pitch,
            "gain": gain,
            "speaker_name": speaker_name,
            "filename": filename,
            "rel_path": rel_path,
            "full_path": full_path,
            "idx": idx + 1
        })

    # 2. Build Rhyming Tasks covering all rhyming words
    for i in range(rhyme_target):
        word = rhyming_words[i % len(rhyming_words)]
        voice_obj = VOICE_PROFILES[i % len(VOICE_PROFILES)]
        voice_id = voice_obj["id"]
        rate = SPEED_RATES[(i // len(VOICE_PROFILES)) % len(SPEED_RATES)]
        pitch = PITCH_ADJUSTMENTS[(i // (len(VOICE_PROFILES) * len(SPEED_RATES))) % len(PITCH_ADJUSTMENTS)]
        gain = VOLUME_GAINS_DB[i % len(VOLUME_GAINS_DB)]
        
        speaker_name = f"synthetic_{voice_id.split('-')[-1]}"
        safe_word = sanitize_for_filename(word)
        filename = f"{speaker_name}_{safe_word}_silent_{i+1:04d}.wav"
        folder = os.path.join("negative_word", "rhyming", voice_id)
        rel_path = os.path.join("dataset", "synthetic", folder, filename).replace("\\", "/")
        full_path = os.path.join(BASE_DIR, rel_path)
        
        tasks_to_run.append({
            "word": word,
            "category": "rhyming_word",
            "folder": folder,
            "voice": voice_id,
            "rate": rate,
            "pitch": pitch,
            "gain": gain,
            "speaker_name": speaker_name,
            "filename": filename,
            "rel_path": rel_path,
            "full_path": full_path,
            "idx": i + 1
        })

    # 3. Build Negative Tasks covering all negative words
    for i in range(negative_target):
        word = negative_words[i % len(negative_words)]
        voice_obj = VOICE_PROFILES[(i + 3) % len(VOICE_PROFILES)]
        voice_id = voice_obj["id"]
        rate = SPEED_RATES[(i + 1) % len(SPEED_RATES)]
        pitch = PITCH_ADJUSTMENTS[(i + 2) % len(PITCH_ADJUSTMENTS)]
        gain = VOLUME_GAINS_DB[i % len(VOLUME_GAINS_DB)]
        
        speaker_name = f"synthetic_{voice_id.split('-')[-1]}"
        safe_word = sanitize_for_filename(word)
        filename = f"{speaker_name}_{safe_word}_silent_{i+1:04d}.wav"
        folder = os.path.join("negative_word", "general", voice_id)
        rel_path = os.path.join("dataset", "synthetic", folder, filename).replace("\\", "/")
        full_path = os.path.join(BASE_DIR, rel_path)
        
        tasks_to_run.append({
            "word": word,
            "category": "general_negative",
            "folder": folder,
            "voice": voice_id,
            "rate": rate,
            "pitch": pitch,
            "gain": gain,
            "speaker_name": speaker_name,
            "filename": filename,
            "rel_path": rel_path,
            "full_path": full_path,
            "idx": i + 1
        })

    total_tasks = len(tasks_to_run)
    print(f"\n[PLAN] Total Planned Synthetic Audio Clips: {total_tasks}")
    print(f" - Keyword '{trigger_word}' clips: {trigger_target}")
    print(f" - Rhyming words ({len(rhyming_words)} unique) clips: {rhyme_target}")
    print(f" - General negative words ({len(negative_words)} unique) clips: {negative_target}")

    if dry_run:
        print("\n[DRY RUN] Completed plan check successfully. No audio generated.")
        return

    # Execute Async TTS Generation
    semaphore = asyncio.Semaphore(concurrency)
    completed_count = 0
    start_time = datetime.datetime.now()

    async def worker(item):
        nonlocal completed_count
        # Skip if already exists and non-empty
        if os.path.exists(item["full_path"]) and os.path.getsize(item["full_path"]) > 1000:
            ok = True
        else:
            ok = await generate_single_tts_clip(
                word=item["word"],
                voice=item["voice"],
                rate=item["rate"],
                pitch=item["pitch"],
                volume_db=item["gain"],
                output_path=item["full_path"],
                semaphore=semaphore
            )
        completed_count += 1
        if completed_count % 50 == 0 or completed_count == total_tasks:
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            rate_sec = completed_count / max(1.0, elapsed)
            print(f"Progress: {completed_count}/{total_tasks} ({completed_count/total_tasks*100:.1f}%) | {rate_sec:.1f} clips/sec")
        
        if ok:
            return item
        return None

    results = await asyncio.gather(*(worker(item) for item in tasks_to_run))
    success_items = [r for r in results if r is not None]

    print(f"\n[COMPLETED] Successfully generated/verified {len(success_items)} / {total_tasks} synthetic audio samples.")

    # SQLite & CSV Logging
    conn = sqlite3.connect(SQLITE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataset_catalog (
            file_id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            speaker_name TEXT NOT NULL,
            word_spoken TEXT NOT NULL,
            category TEXT NOT NULL,
            environment TEXT NOT NULL,
            duration_ms INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    
    csv_file = open(CSV_PATH, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for item in success_items:
        file_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT OR REPLACE INTO dataset_catalog 
            (file_id, file_path, speaker_name, word_spoken, category, environment, duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            file_id,
            item["rel_path"],
            item["speaker_name"],
            item["word"],
            item["category"],
            "silent_room",
            1000,
            now_iso
        ))
        csv_writer.writerow([
            file_id,
            item["rel_path"],
            item["speaker_name"],
            item["word"],
            item["category"],
            "silent_room",
            1000,
            now_iso
        ])

    conn.commit()
    conn.close()
    csv_file.close()

    print(f"[METADATA] Logged {len(success_items)} entries into {SQLITE_PATH} and {CSV_PATH}")

def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic TTS Dataset for Keyword Spotting")
    parser.add_argument("--trigger-count", type=int, default=500, help="Target keyword sample count (default: 500)")
    parser.add_argument("--rhyme-count", type=int, default=250, help="Target rhyming sample count (default: 250)")
    parser.add_argument("--negative-count", type=int, default=250, help="Target negative sample count (default: 250)")
    parser.add_argument("--concurrency", type=int, default=12, help="Max concurrent async TTS streams (default: 12)")
    parser.add_argument("--dry-run", action="store_true", help="Preview generation matrix without creating files")
    args = parser.parse_args()

    asyncio.run(run_synthetic_pipeline(
        trigger_target=args.trigger_count,
        rhyme_target=args.rhyme_count,
        negative_target=args.negative_count,
        concurrency=args.concurrency,
        dry_run=args.dry_run
    ))

if __name__ == "__main__":
    main()
