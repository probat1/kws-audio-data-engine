"""
Synthetic TTS Dataset Generation Runner
Delegates to the modular tts_pipeline package.
"""

import sys
from tts_pipeline.generate_synthetic_data import main, run_synthetic_pipeline, VOICE_PROFILES, SPEED_RATES, PITCH_ADJUSTMENTS, process_and_enforce_one_second_wav

if __name__ == "__main__":
    main()
