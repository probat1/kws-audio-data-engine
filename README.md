# Voice Data Engineering & Acoustic Augmentation Suite

A specialized, production-ready audio data engineering pipeline for **Keyword-Spotting (KWS) Machine Learning models**. Built strictly adhering to the **Data Engineering & Collection Tooling Lead Specification**.

---

## 🌟 Architecture Overview

```
                          ┌────────────────────────┐
                          │   words_config.json    │
                          └───────────┬────────────┘
                                      ▼
                        [ tts_pipeline/ ]
             (Edge-TTS, 10+ Voices, Pitch/Rate Matrix)
                                      │
                                      ▼
                      [ dataset/synthetic/ ]
                                      │
                                      ▼
                       [ noise_pipeline/ ]
             (Dynamic SNR Mixing: Clean, 15dB, 10dB, 5dB)
             (Noise Banks: Claps, Fan Hum, Room Chatter)
                                      │
                                      ▼
                     [ dataset/final_train/ for ML Lead ]
```

---

## 📂 Project Structure

```
Voice-data-collector/
│
├── 🗣️ tts_pipeline/                     # Synthetic Voice Generation Package (Sprint 2)
│   ├── __init__.py
│   ├── voices_config.py                 # Vocal profiles, pitch, speed, and volume matrix
│   ├── tts_engine.py                    # Edge-TTS streamer, audio trimming & 1.0s normalization
│   └── generate_synthetic_data.py       # Batch orchestrator & CLI parser
│
├── 🔊 noise_pipeline/                   # Noise Merging & Augmentation Package (Sprint 3)
│   ├── __init__.py
│   ├── snr_mixer.py                     # RMS energy calculation & Dynamic SNR mixing engine
│   ├── download_noise_banks.py          # Acoustic noise generator & downloader (claps, fan, chatter)
│   └── augment_dataset.py               # Dataset scanner, batch 4-way augmentor & packager
│
├── 📂 dataset/                          # Audio Datasets & Catalogs
│   ├── synthetic/                       # 16 kHz 1.0s Mono synthetic TTS samples
│   ├── noise_banks/                     # Acoustic noise banks (fan, chatter, claps, knocks)
│   └── final_train/                     # Balanced, 4-way augmented dataset for ML training
│       ├── trigger_word/
│       ├── negative_word/rhyming/
│       ├── negative_word/general/
│       └── train_catalog.csv
│
├── words_config.json                    # Target keyword, rhyming, and negative vocabulary config
├── generate_synthetic_data.py           # Convenience root CLI runner for TTS
├── download_noise_banks.py              # Convenience root CLI runner for Noise Banks
├── augment_dataset.py                   # Convenience root CLI runner for Noise Augmentation
├── test_pipeline.py                     # Automated test suite (pytest)
├── metadata.db                          # SQLite catalog database
├── dataset_catalog.csv                  # Raw dataset CSV catalog
└── requirements.txt                     # Project dependencies
```

---

## 🚀 Execution Guide

### 1. Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

### 2. Synthetic TTS Generation (`tts_pipeline`)

Generate diverse synthetic vocal profiles across pitch, rate, and volume variations:

```bash
# Preview plan without generating files
python -m tts_pipeline.generate_synthetic_data --dry-run

# Generate 1,500 keyword samples + 1,000 rhyming/negative samples
python -m tts_pipeline.generate_synthetic_data --trigger-count 1500 --other-count 1000 --concurrency 10
```

- **Voice Profiles**: 10+ distinct voices in `tts_pipeline/voices_config.py` (Indian English, Hindi, US, UK, AU, CA, Male & Female).
- **Matrix Variations**: 5 speed rates (0.75x–1.25x), 5 pitch adjustments (-15Hz to +20Hz), 3 volume levels (-6dB, 0dB, +4dB).
- **Auto-trimming**: Strictly enforces **1.0-second 16 kHz Mono WAV** files with centered speech.

---

### 3. Noise Augmentation & Packaging (`noise_pipeline`)

#### Step A: Generate Acoustic Noise Banks
```bash
python -m noise_pipeline.download_noise_banks
```
Generates 16 kHz noise profiles in `dataset/noise_banks/`:
- Low Noise: Fan hum & motor airflow
- Medium Noise: Room background chatter & keyboard typing
- High Noise / Impulse: Sudden hand claps (for the **"Clap" Test**) & desk/door knocks

#### Step B: Run Dynamic SNR Augmentation
```bash
python -m noise_pipeline.augment_dataset
```
Generates **4 variations per clean recording** at exact Signal-to-Noise Ratios:
1. **Clean Sample (Original)**: 100% clean voice.
2. **Low Noise (15 dB SNR)**: Subtle background fan hum.
3. **Medium Noise (10 dB SNR)**: Ambient room chatter / typing.
4. **High Noise / Impulse (5 dB SNR)**: Sudden transient claps or knocks mixed in.

Outputs the final training dataset into `dataset/final_train/` with `dataset/final_train/train_catalog.csv` and SQLite table `final_train_catalog`.

---

## 🧪 Testing

Run the automated test suite:
```bash
pytest -v
```
