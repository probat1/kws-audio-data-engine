"""
Acoustic Noise Generator and Downloader
Prepares standard background noises and impulse sounds in dataset/noise_banks/
"""

import os
import numpy as np
import soundfile as sf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOISE_DIR = os.path.join(BASE_DIR, "dataset", "noise_banks")
os.makedirs(NOISE_DIR, exist_ok=True)

def generate_fan_hum(duration_sec=10.0, sr=16000) -> np.ndarray:
    """Generates low-frequency fan/motor hum with harmonic hums and broadband airflow."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    hum = 0.4 * np.sin(2 * np.pi * 60 * t) + 0.3 * np.sin(2 * np.pi * 120 * t) + 0.15 * np.sin(2 * np.pi * 180 * t)
    noise = np.random.normal(0, 0.08, len(t))
    b, a = [0.05], [1.0, -0.95]
    for i in range(1, len(noise)):
        noise[i] = a[1] * -noise[i-1] + b[0] * noise[i]
    combined = hum + noise
    return (combined / np.max(np.abs(combined))) * 0.8

def generate_hand_clap_transients(duration_sec=10.0, sr=16000) -> np.ndarray:
    """Generates sharp, high-frequency hand clap impulses (Clap Test)."""
    samples = int(sr * duration_sec)
    audio = np.zeros(samples, dtype=np.float32)
    
    clap_times = [1.2, 3.5, 5.8, 8.1]
    for ct in clap_times:
        start_idx = int(ct * sr)
        if start_idx >= samples - 4000:
            continue
        decay_samples = int(sr * 0.15)
        t_decay = np.linspace(0, 0.15, decay_samples)
        env = np.exp(-t_decay * 40)
        burst = (
            0.6 * np.random.normal(0, 1, decay_samples) +
            0.3 * np.sin(2 * np.pi * 2400 * t_decay) +
            0.2 * np.sin(2 * np.pi * 3600 * t_decay)
        )
        audio[start_idx : start_idx + decay_samples] += (burst * env)
    
    audio += np.random.normal(0, 0.01, samples)
    return (audio / np.max(np.abs(audio))) * 0.9

def generate_room_chatter_typing(duration_sec=10.0, sr=16000) -> np.ndarray:
    """Generates ambient room background chatter, clicks, and keyboard typing."""
    samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, samples, endpoint=False)
    
    chatter = 0.05 * np.sin(2 * np.pi * 220 * t) + 0.04 * np.sin(2 * np.pi * 440 * t) + 0.03 * np.sin(2 * np.pi * 780 * t)
    noise = np.random.normal(0, 0.03, samples)
    
    np.random.seed(42)
    click_indices = np.random.choice(range(1000, samples - 1000), size=18, replace=False)
    for idx in click_indices:
        click_len = int(sr * 0.015)
        t_click = np.linspace(0, 0.015, click_len)
        click_env = np.exp(-t_click * 300)
        click = np.sin(2 * np.pi * 3200 * t_click) * click_env * 0.4
        chatter[idx : idx + click_len] += click

    combined = chatter + noise
    return (combined / np.max(np.abs(combined))) * 0.85

def generate_door_knocks_taps(duration_sec=10.0, sr=16000) -> np.ndarray:
    """Generates wooden door knocks and desk taps."""
    samples = int(sr * duration_sec)
    audio = np.zeros(samples, dtype=np.float32)
    
    knock_times = [0.8, 1.0, 1.2, 4.0, 4.2, 7.5, 7.7, 7.9]
    for kt in knock_times:
        start_idx = int(kt * sr)
        if start_idx >= samples - 3000:
            continue
        knock_len = int(sr * 0.08)
        t_knock = np.linspace(0, 0.08, knock_len)
        env = np.exp(-t_knock * 80)
        knock = (np.sin(2 * np.pi * 280 * t_knock) * 0.7 + np.sin(2 * np.pi * 560 * t_knock) * 0.3) * env
        audio[start_idx : start_idx + knock_len] += knock
        
    audio += np.random.normal(0, 0.005, samples)
    return (audio / np.max(np.abs(audio))) * 0.9

def generate_white_pink_noise(duration_sec=10.0, sr=16000) -> np.ndarray:
    """Generates standard acoustic white/pink noise."""
    samples = int(sr * duration_sec)
    white = np.random.normal(0, 0.1, samples)
    return (white / np.max(np.abs(white))) * 0.75

def prepare_noise_banks():
    print("=" * 60)
    print("Preparing Noise Banks for Acoustic Augmentation Pipeline")
    print(f"Destination: {NOISE_DIR}")
    print("=" * 60)
    
    sr = 16000
    noise_profiles = [
        ("fan_hum_motor.wav", generate_fan_hum(10.0, sr), "Low Noise (Fan hum @ 15dB SNR)"),
        ("ambient_room_chatter_typing.wav", generate_room_chatter_typing(10.0, sr), "Medium Noise (Room chatter @ 10dB SNR)"),
        ("sudden_hand_claps.wav", generate_hand_clap_transients(10.0, sr), "High Noise / Impulse (Sudden claps @ 5dB SNR)"),
        ("door_knocks_desk_taps.wav", generate_door_knocks_taps(10.0, sr), "High Noise / Impulse (Door knocks @ 5dB SNR)"),
        ("white_acoustic_noise.wav", generate_white_pink_noise(10.0, sr), "Broadband Noise")
    ]
    
    for filename, audio_arr, desc in noise_profiles:
        filepath = os.path.join(NOISE_DIR, filename)
        sf.write(filepath, audio_arr, sr, subtype="PCM_16", format="WAV")
        print(f" [OK] Created: {filename:<35} | {desc}")
    
    print("\nAll noise banks ready for dynamic SNR augmentation.")

if __name__ == "__main__":
    prepare_noise_banks()
