"""
Automated Test Suite for Voice Data Engineering Pipelines
Tests:
- Sprint 2 (tts_pipeline): Parameter matrices, 1.0s WAV audio format enforcement, Edge-TTS streamer
- Sprint 3 (noise_pipeline): Dynamic SNR noise mixing, acoustic generators, dataset packaging
"""

import os
import io
import pytest
import sqlite3
import numpy as np
import soundfile as sf

from tts_pipeline import process_and_enforce_one_second_wav, VOICE_PROFILES, SPEED_RATES, PITCH_ADJUSTMENTS, VOLUME_GAINS_DB
from noise_pipeline import mix_audio_snr, calculate_rms, overlay_noise
from noise_pipeline.augment_dataset import discover_clean_audio_files
from noise_pipeline.download_noise_banks import generate_fan_hum, generate_hand_clap_transients

# ----------------------------------------------------
# Sprint 2 Tests (tts_pipeline)
# ----------------------------------------------------
def test_synthetic_parameter_matrix():
    assert len(VOICE_PROFILES) >= 10
    assert len(SPEED_RATES) >= 5
    assert len(PITCH_ADJUSTMENTS) >= 5
    assert len(VOLUME_GAINS_DB) >= 3

def test_synthetic_1s_enforcement():
    sr = 16000
    t = np.linspace(0, 0.4, int(sr * 0.4))
    pulse = np.sin(2 * np.pi * 300 * t).astype(np.float32)
    
    buf = io.BytesIO()
    sf.write(buf, pulse, sr, subtype="PCM_16", format="WAV")
    
    wav_1s = process_and_enforce_one_second_wav(buf.getvalue(), volume_gain_db=0.0)
    data, sample_rate = sf.read(io.BytesIO(wav_1s))
    
    assert sample_rate == 16000
    assert len(data) == 16000
    assert len(data.shape) == 1

# ----------------------------------------------------
# Sprint 3 Tests (noise_pipeline)
# ----------------------------------------------------
def test_noise_generators():
    fan = generate_fan_hum(duration_sec=1.0, sr=16000)
    claps = generate_hand_clap_transients(duration_sec=1.0, sr=16000)
    
    assert len(fan) == 16000
    assert len(claps) == 16000
    assert calculate_rms(fan) > 0
    assert calculate_rms(claps) > 0

def test_dynamic_snr_mixing():
    sr = 16000
    speech = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
    noise = np.random.normal(0, 0.5, sr).astype(np.float32)
    
    mixed_15db = mix_audio_snr(speech, noise, target_snr_db=15.0)
    assert len(mixed_15db) == sr
    assert np.max(np.abs(mixed_15db)) <= 1.0

    mixed_5db = mix_audio_snr(speech, noise, target_snr_db=5.0)
    assert len(mixed_5db) == sr
    assert np.max(np.abs(mixed_5db)) <= 1.0
