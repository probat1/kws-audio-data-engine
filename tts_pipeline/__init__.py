"""
TTS Pipeline Package for Synthetic Voice Generation
"""

from .voices_config import VOICE_PROFILES, SPEED_RATES, PITCH_ADJUSTMENTS, VOLUME_GAINS_DB
from .tts_engine import process_and_enforce_one_second_wav, generate_single_tts_clip

__all__ = [
    "VOICE_PROFILES",
    "SPEED_RATES",
    "PITCH_ADJUSTMENTS",
    "VOLUME_GAINS_DB",
    "process_and_enforce_one_second_wav",
    "generate_single_tts_clip"
]
