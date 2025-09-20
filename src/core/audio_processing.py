import torch
import torchaudio
import whisper
import io
import time
import uuid
from typing import BinaryIO, Dict, Any, Tuple
from demucs.apply import apply_model
from demucs.pretrained import get_model as get_demucs_model

# --- Model Loading ---
# We load models once to avoid reloading them on every request.
# This is a simple approach for a single-worker setup.
# For multi-worker setups (gunicorn), this would be per-process.

print("Loading ML models...")
# Auto-detect device (GPU if available, otherwise CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load Demucs for vocal separation
DEMUCS_MODEL = get_demucs_model('htdemucs')
DEMUCS_MODEL.to(DEVICE)
print("Demucs model loaded.")

# Load Whisper for transcription
# Using 'small' as a good balance of speed and accuracy.
WHISPER_MODEL = whisper.load_model("small", device=DEVICE)
print("Whisper model loaded.")
print("--- Models are ready ---")


def load_audio(file: BinaryIO, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    Loads an audio file into a tensor, resamples it, and converts to mono.

    Args:
        file: The audio file object.
        target_sr: The target sample rate.

    Returns:
        A tuple containing the audio tensor and the original sample rate.
    """
    try:
        waveform, original_sr = torchaudio.load(file)
        waveform = waveform.to(DEVICE)

        # Resample if necessary
        if original_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr).to(DEVICE)
            waveform = resampler(waveform)

        # Convert to mono by averaging channels
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform, target_sr
    except Exception as e:
        # This could catch errors from unsupported formats, corrupted files, etc.
        print(f"Error loading audio: {e}")
        raise ValueError("Failed to load or decode the audio file.")


def separate_vocals(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Separates vocals from a given audio waveform using Demucs.

    Args:
        waveform: The input audio tensor.
        sample_rate: The sample rate of the audio.

    Returns:
        The audio tensor for the isolated vocal track.
    """
    # Demucs expects a waveform of shape [batch, channels, samples]
    # Our waveform is [1, samples], so we add a batch dimension if needed.
    if waveform.dim() == 2:
         waveform = waveform.unsqueeze(0) # Becomes [1, 1, samples] which is not right
         waveform = waveform.squeeze(1) # Back to [1, samples]

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0) # [1, samples]

    # Demucs model works with its own sample rate, we handle it internally
    demucs_sr = DEMUCS_MODEL.samplerate
    resampler_to_demucs = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=demucs_sr).to(DEVICE)
    resampler_from_demucs = torchaudio.transforms.Resample(orig_freq=demucs_sr, new_freq=sample_rate).to(DEVICE)

    # Resample to what demucs expects
    resampled_waveform = resampler_to_demucs(waveform)
    
    # The model returns a dictionary of sources
    # Shape: [batch, sources, channels, samples]
    sources = apply_model(DEMUCS_MODEL, resampled_waveform.unsqueeze(1), device=DEVICE, progress=True)[0]
    
    # Vocals are at index 3 for the htdemucs model
    vocals = sources[3]

    # Resample back to the original target sample rate
    vocals_resampled = resampler_from_demucs(vocals)

    return vocals_resampled


def transcribe_audio(waveform: torch.Tensor) -> Dict[str, Any]:
    """
    Transcribes the given audio waveform using Whisper.

    Args:
        waveform: The audio tensor of the vocals.

    Returns:
        The transcription result from Whisper.
    """
    # Whisper expects a single-channel waveform (not in a batch)
    waveform_squeezed = waveform.squeeze(0).cpu()
    
    result = WHISPER_MODEL.transcribe(waveform_squeezed, fp16=False) # fp16=False for CPU
    return result


def process_audio_pipeline(
    file: BinaryIO,
    enable_separation: bool,
    language_hint: str = None
) -> Dict[str, Any]:
    """
    Orchestrates the full audio processing pipeline.

    Args:
        file: The audio file to process.
        enable_separation: Flag to enable/disable vocal separation.
        language_hint: Optional language hint for Whisper.

    Returns:
        A dictionary containing the structured transcription result.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    timings = {}

    # 1. Load and Normalize Audio
    load_start = time.time()
    try:
        waveform, sample_rate = load_audio(file)
    except ValueError as e:
        raise e # Re-raise the specific error to be caught by the API
    duration_sec = waveform.shape[-1] / sample_rate
    timings["load"] = int((time.time() - load_start) * 1000)

    # 2. Vocal Separation (if enabled)
    separation_start = time.time()
    pipeline_info = {
        "separation": {"enabled": enable_separation, "method": "none"},
        "transcription": {"model": "whisper-small"}
    }
    if enable_separation:
        waveform = separate_vocals(waveform, sample_rate)
        pipeline_info["separation"]["method"] = "demucs"
    timings["separation"] = int((time.time() - separation_start) * 1000)

    # 3. Transcription
    transcription_start = time.time()
    transcription_result = transcribe_audio(waveform)
    timings["transcription"] = int((time.time() - transcription_start) * 1000)

    total_time = time.time() - start_time
    timings["total"] = int(total_time * 1000)

    # 4. Format Response
    response = {
        "request_id": request_id,
        "duration_sec": round(duration_sec, 2),
        "sample_rate": sample_rate,
        "pipeline": pipeline_info,
        "segments": [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in transcription_result["segments"]
        ],
        "text": transcription_result["text"],
        "language": transcription_result["language"],
        "timings_ms": timings
    }

    return response