"""
Audio Noise Augmentation & Dataset Packaging Runner
Delegates to the modular noise_pipeline package.
"""

from noise_pipeline.augment_dataset import main, run_augmentation_pipeline, discover_clean_audio_files
from noise_pipeline.snr_mixer import calculate_rms, mix_audio_snr, overlay_noise

if __name__ == "__main__":
    main()
