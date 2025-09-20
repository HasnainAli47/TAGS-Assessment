# This script is intended to be run during the Docker build process
# to pre-download and cache the necessary models.

from demucs.pretrained import get_model as get_demucs_model
import whisper

if __name__ == "__main__":
    print("Downloading and caching Demucs model...")
    get_demucs_model('htdemucs')
    print("Demucs model is cached.")

    print("\nDownloading and caching Whisper model (small)...")
    whisper.load_model("small")
    print("Whisper model is cached.")

    print("\nAll models have been downloaded and are ready for offline use.")