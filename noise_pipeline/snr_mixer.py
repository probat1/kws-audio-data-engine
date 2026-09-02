"""
Mathematical RMS Energy & Dynamic SNR Audio Overlay Engine
"""

import os
import random
import numpy as np
import soundfile as sf

def calculate_rms(audio_arr: np.ndarray) -> float:
    """Calculates Root-Mean-Square (RMS) amplitude of audio array."""
    if len(audio_arr) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_arr ** 2)))

def mix_audio_snr(clean_arr: np.ndarray, noise_arr: np.ndarray, target_snr_db: float) -> np.ndarray:
    """
    Overlays noise onto clean speech at a specific Signal-to-Noise Ratio (SNR) in dB.
    Formula: SNR_dB = 20 * log10(RMS_speech / RMS_noise)
    => Required Noise Scale Factor = RMS_speech / (10^(SNR/20) * RMS_noise)
    """
    clean_len = len(clean_arr)
    noise_len = len(noise_arr)

    if noise_len < clean_len:
        repeats = int(np.ceil(clean_len / noise_len))
        noise_arr = np.tile(noise_arr, repeats)[:clean_len]
    elif noise_len > clean_len:
        start_idx = random.randint(0, noise_len - clean_len)
        noise_arr = noise_arr[start_idx : start_idx + clean_len]

    speech_rms = calculate_rms(clean_arr)
    noise_rms = calculate_rms(noise_arr)

    if speech_rms == 0 or noise_rms == 0:
        return clean_arr

    desired_noise_rms = speech_rms / (10.0 ** (target_snr_db / 20.0))
    scale_factor = desired_noise_rms / noise_rms

    scaled_noise = noise_arr * scale_factor
    mixed = clean_arr + scaled_noise

    # Peak normalize to 0.95 to eliminate any potential digital clipping
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = (mixed / peak) * 0.95

    return mixed

def overlay_noise(clean_path: str, noise_path: str, output_path: str, snr_gain: int = 10, sr: int = 16000):
    """
    Overlays background noise onto clean voice audio using high-precision NumPy/Soundfile.
    """
    clean_data, clean_sr = sf.read(clean_path)
    noise_data, noise_sr = sf.read(noise_path)

    if len(clean_data.shape) > 1:
        clean_data = np.mean(clean_data, axis=1)
    if len(noise_data.shape) > 1:
        noise_data = np.mean(noise_data, axis=1)

    augmented = mix_audio_snr(clean_data, noise_data, target_snr_db=float(snr_gain))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, augmented, sr, subtype="PCM_16", format="WAV")
