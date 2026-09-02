"""
Noise Banks Downloader and Generator Runner
Delegates to the modular noise_pipeline package.
"""

from noise_pipeline.download_noise_banks import prepare_noise_banks, generate_fan_hum, generate_hand_clap_transients, generate_room_chatter_typing, generate_door_knocks_taps, generate_white_pink_noise

if __name__ == "__main__":
    prepare_noise_banks()
