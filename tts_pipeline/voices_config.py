"""
Vocal Profiles, Speed, Pitch, and Volume Matrices for Synthetic TTS
"""

VOICE_PROFILES = [
    # Indian Accents & Subcontinent
    {"id": "en-IN-NeerjaNeural", "gender": "female", "accent": "indian_english"},
    {"id": "en-IN-PrabhatNeural", "gender": "male", "accent": "indian_english"},
    {"id": "hi-IN-SwaraNeural", "gender": "female", "accent": "hindi"},
    {"id": "hi-IN-MadhurNeural", "gender": "male", "accent": "hindi"},
    # US English
    {"id": "en-US-GuyNeural", "gender": "male", "accent": "us_english"},
    {"id": "en-US-JennyNeural", "gender": "female", "accent": "us_english"},
    {"id": "en-US-AriaNeural", "gender": "female", "accent": "us_english"},
    {"id": "en-US-ChristopherNeural", "gender": "male", "accent": "us_english"},
    # UK English
    {"id": "en-GB-SoniaNeural", "gender": "female", "accent": "uk_english"},
    {"id": "en-GB-RyanNeural", "gender": "male", "accent": "uk_english"},
    # Global Accents
    {"id": "en-AU-NatashaNeural", "gender": "female", "accent": "australian"},
    {"id": "en-CA-LiamNeural", "gender": "male", "accent": "canadian"}
]

SPEED_RATES = ["-25%", "-10%", "+0%", "+10%", "+25%"] # 0.75x, 0.90x, 1.0x, 1.10x, 1.25x
PITCH_ADJUSTMENTS = ["-15Hz", "-5Hz", "+0Hz", "+10Hz", "+20Hz"]
VOLUME_GAINS_DB = [-6.0, 0.0, 4.0] # dB gains applied in post-processing
