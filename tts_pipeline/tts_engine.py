"""
TTS Generation Engine and Audio Formatting Utilities
"""

import os
import io
import asyncio
import edge_tts
import numpy as np
import scipy.signal
import soundfile as sf

def process_and_enforce_one_second_wav(raw_audio_bytes: bytes, volume_gain_db: float = 0.0) -> bytes:
    """
    Decodes audio stream, trims leading/trailing silence, applies volume gain,
    pads/centers or trims to exact 1.0 second (16000 samples) at 16 kHz Mono PCM_16 WAV.
    """
    audio_data, sr = sf.read(io.BytesIO(raw_audio_bytes))
    
    # Convert to mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Resample to 16 kHz
    target_sr = 16000
    if sr != target_sr:
        num_samples = int(len(audio_data) * target_sr / sr)
        audio_data = scipy.signal.resample(audio_data, num_samples)
    
    # Apply volume gain (dB to linear scale)
    if volume_gain_db != 0.0:
        gain_factor = 10.0 ** (volume_gain_db / 20.0)
        audio_data = audio_data * gain_factor

    # Auto-trim leading & trailing silence (threshold: 8% of peak amplitude or min 0.015)
    abs_audio = np.abs(audio_data)
    peak = np.max(abs_audio) if len(abs_audio) > 0 else 0
    if peak > 0:
        threshold = max(0.015, peak * 0.08)
        non_silent = np.where(abs_audio >= threshold)[0]
        if len(non_silent) > 0:
            start_idx = max(0, non_silent[0] - int(target_sr * 0.05)) # 50ms pre-padding
            end_idx = min(len(audio_data), non_silent[-1] + int(target_sr * 0.05)) # 50ms post-padding
            audio_data = audio_data[start_idx:end_idx]

    # Enforce strict 1.0-second length (16000 samples)
    target_length = 16000 # 1.0s at 16 kHz
    current_length = len(audio_data)
    
    if current_length < target_length:
        pad_total = target_length - current_length
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        audio_data = np.pad(audio_data, (pad_left, pad_right), mode="constant", constant_values=0)
    elif current_length > target_length:
        excess = current_length - target_length
        start_crop = excess // 2
        audio_data = audio_data[start_crop : start_crop + target_length]

    # Peak normalization to ~0.95
    max_amp = np.max(np.abs(audio_data))
    if max_amp > 0:
        audio_data = (audio_data / max_amp) * 0.95
    
    # Output to 16-bit PCM WAV
    out_buf = io.BytesIO()
    sf.write(out_buf, audio_data, target_sr, subtype="PCM_16", format="WAV")
    return out_buf.getvalue()

async def generate_single_tts_clip(
    word: str,
    voice: str,
    rate: str,
    pitch: str,
    volume_db: float,
    output_path: str,
    semaphore: asyncio.Semaphore,
    retries: int = 3
) -> bool:
    """Generates a single TTS audio clip with retry logic."""
    async with semaphore:
        for attempt in range(retries):
            try:
                communicate = edge_tts.Communicate(
                    text=word,
                    voice=voice,
                    rate=rate,
                    pitch=pitch
                )
                
                audio_stream = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_stream += chunk["data"]
                
                if len(audio_stream) < 200:
                    raise ValueError("Received empty or incomplete audio chunk from TTS engine.")
                
                wav_1s = process_and_enforce_one_second_wav(audio_stream, volume_gain_db=volume_db)
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(wav_1s)
                
                return True
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    print(f"[ERROR] Failed generating '{word}' with voice {voice} (rate={rate}, pitch={pitch}): {e}")
                    return False
    return False
